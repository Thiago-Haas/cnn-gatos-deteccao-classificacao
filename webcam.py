"""Detecta gatos e classifica suas raças usando um checkpoint do notebook."""

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from ultralytics import YOLO

from cat_landmarks import detect_landmarks, draw_landmarks, load_landmark_model


PROJECT_DIR = Path(__file__).resolve().parent
WINDOW = "Gatos - Q ou Esc para sair"


def load_classifier(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Modelo não encontrado: {path}. Copie o .pt exportado do Colab "
            "para artifacts/ ou informe --model."
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    labels = {int(i): name for i, name in checkpoint["idx_to_breed"].items()}
    if sorted(labels) != list(range(len(labels))) or not labels:
        raise ValueError("O checkpoint precisa mapear índices consecutivos para as raças.")

    # O notebook original usa Linear; o corrigido usa Dropout + Linear.
    model = models.resnet50(weights=None)
    linear = nn.Linear(model.fc.in_features, len(labels))
    if "fc.1.weight" in state:
        model.fc = nn.Sequential(nn.Dropout(p=0.4), linear)
    elif "fc.weight" in state:
        model.fc = linear
    else:
        raise ValueError("Camada classificadora incompatível com os notebooks do projeto.")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), labels


def preprocessing(image_size=224):
    # Mesmas transformações de avaliação do notebook, sobre uma imagem PIL RGB.
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


BREED_CONF_THRESHOLD = 0.40  # Confiança mínima para identificar a raça; abaixo disso: SRD.


@torch.inference_mode()
def annotate_frame(frame, detector, classifier, labels, transform, device, conf,
                   landmark_model=None, landmark_conf=0.25, breed_conf=BREED_CONF_THRESHOLD):
    result = detector.predict(
        source=frame, classes=[15], conf=conf, device=str(device), verbose=False,
    )[0]
    annotated = frame.copy()
    found = False
    height, width = frame.shape[:2]
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = Image.fromarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        # Estima os pontos antes da raça; a ResNet recebe o recorte RGB original.
        if landmark_model is not None:
            features = detect_landmarks(crop, landmark_model, device, conf=landmark_conf)
            annotated = draw_landmarks(annotated, features, offset=(x1, y1))
        tensor = transform(crop).unsqueeze(0).to(device)
        probabilities = classifier(tensor).softmax(dim=1)[0]
        index = int(probabilities.argmax().item())
        top_prob = probabilities[index].item()
        if top_prob >= breed_conf:
            breed_name = labels[index].replace("_", " ")
            text = f"{breed_name} {top_prob:.1%}"
            color = (0, 255, 0)
        else:
            text = f"Sem Raça Definida ({top_prob:.1%})"
            color = (0, 165, 255)  # laranja: indica baixa confiança
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, text, (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        found = True
    if not found:
        cv2.putText(annotated, "Nenhum gato detectado", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return annotated


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=PROJECT_DIR / "artifacts" / "cat_breed_classifier.pt")
    parser.add_argument("--yolo", type=Path, default=PROJECT_DIR / "yolov8n.pt",
                        help="Pesos YOLO COCO (baixados automaticamente se ausentes).")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0).")
    parser.add_argument("--landmarks", type=Path,
                        help="Modelo CAT de nove pontos faciais (artifacts/cat_landmarks.pt).")
    parser.add_argument("--landmark-conf", type=float, default=0.25,
                        help="Confiança mínima da detecção da cabeça.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--conf", type=float, default=0.25, help="Confiança mínima da detecção.")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Resolução usada no treino do classificador (padrão: 224).")
    parser.add_argument("--breed-conf", type=float, default=BREED_CONF_THRESHOLD,
                        help="Confiança mínima para identificar a raça (padrão: 0.40). "
                             "Abaixo desse valor o gato é exibido como 'Sem Raça Definida'.")
    args = parser.parse_args()
    if not 0 < args.conf <= 1:
        parser.error("--conf deve estar entre 0 (exclusivo) e 1.")
    if not 0 < args.landmark_conf <= 1:
        parser.error("--landmark-conf deve estar entre 0 (exclusivo) e 1.")
    if not 0 < args.breed_conf <= 1:
        parser.error("--breed-conf deve estar entre 0 (exclusivo) e 1.")
    if args.image_size <= 0:
        parser.error("--image-size deve ser positivo.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA indisponível; use --device cpu ou auto.")
    return args


def main():
    args = parse_args()
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    classifier, labels = load_classifier(args.model, device)
    detector = YOLO(str(args.yolo))
    landmark_model = load_landmark_model(args.landmarks) if args.landmarks else None
    transform = preprocessing(args.image_size)
    print(f"Modelo carregado: {len(labels)} raças | dispositivo: {device}")
    print("Na janela da webcam, pressione Q ou Esc para sair.")
    capture = cv2.VideoCapture(args.camera)
    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a webcam {args.camera}. "
                "Verifique a conexão, as permissões ou tente --camera 1."
            )
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Não foi possível ler um frame da webcam.")
            annotated = annotate_frame(
                frame, detector, classifier, labels, transform, device, args.conf,
                landmark_model=landmark_model, landmark_conf=args.landmark_conf,
                breed_conf=args.breed_conf,
            )
            cv2.imshow(WINDOW, annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, KeyError, cv2.error) as exc:
        raise SystemExit(f"Erro: {exc}") from exc
