"""Pontos faciais do CAT Dataset: preparação, treinamento e inferência.

Zhang, W.; Sun, J.; Tang, X. Cat Head Detection — How to Effectively Exploit
Shape and Texture Features. ECCV, 2008. Espelho: kaggle.com/datasets/crawford/cat-dataset.
"""

import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml


KEYPOINT_NAMES = (
    "left_eye", "right_eye", "mouth",
    "left_ear_1", "left_ear_2", "left_ear_3",
    "right_ear_1", "right_ear_2", "right_ear_3",
)
CAT_URL = "https://www.kaggle.com/api/v1/datasets/download/crawford/cat-dataset?datasetVersionNumber=2"
SOURCE = "crawford/cat-dataset/versions/2"


def _make_yolo(path):
    # Mantém a preparação independente dos efeitos globais da importação Ultralytics.
    from ultralytics import YOLO
    return YOLO(path)


def download_landmarks(data_dir, archive_path=None):
    """Baixa a versão 2; aceita também o ZIP local baixado no Kaggle."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / "cat_landmarks_raw_v2"
    if (destination / ".complete").is_file():
        return destination
    if destination.exists():
        raise ValueError(f"Download incompleto em {destination}; use outra pasta ou remova a cópia incompleta.")
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staging = Path(temporary)
        if archive_path is None:
            archive_path = staging / "cat.zip"
            print("Baixando CAT Dataset v2 (download de vários GB)...", flush=True)
            with urlopen(CAT_URL, timeout=120) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        extracted = staging / "raw"
        extracted.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    path = Path(member.filename)
                    if path.is_absolute() or ".." in path.parts:
                        raise ValueError(f"Caminho inválido no ZIP: {member.filename}")
                archive.extractall(extracted)
        except zipfile.BadZipFile as exc:
            raise ValueError("O download não é um ZIP. Baixe pelo Kaggle e informe --archive caminho.zip.") from exc
        if not any(extracted.rglob("*.cat")):
            raise ValueError("Nenhuma anotação .cat encontrada no arquivo.")
        (extracted / ".complete").write_text(SOURCE + "\n")
        extracted.rename(destination)
    return destination


def read_annotation(path, width, height):
    """Converte '9 x1 y1 ... x9 y9' para uma linha YOLO pose normalizada."""
    values = np.asarray(Path(path).read_text().split(), dtype=float)
    if len(values) != 19 or values[0] != 9 or not np.isfinite(values).all():
        raise ValueError("Esperados 9 pontos e 18 coordenadas finitas.")
    xy = values[1:].reshape(9, 2)
    visible = ((xy >= 0) & (xy < [width, height])).all(axis=1)
    if visible.sum() < 3:
        raise ValueError("Menos de três pontos dentro da imagem.")
    low, high = xy[visible].min(axis=0), xy[visible].max(axis=0)
    # O dataset fornece pontos, não caixas: derivamos uma caixa da cabeça com margem.
    margin = np.maximum((high - low) * 0.15, 2)
    low = np.maximum(low - margin, 0)
    high = np.minimum(high + margin, [width, height])
    size = np.asarray([width, height])
    box = np.concatenate(((low + high) / (2 * size), (high - low) / size))
    keypoints = np.zeros((9, 3))
    keypoints[visible, :2] = xy[visible] / size
    keypoints[visible, 2] = 2  # ponto anotado; fora da imagem -> não supervisionado
    return np.concatenate(([0], box, keypoints.ravel()))


def prepare_landmarks(raw_dir, output_dir, seed=42):
    """Converte .cat para YOLO pose; agrupa cópias/variantes da mesma foto no split."""
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()
    config = {"source": SOURCE, "raw_dir": str(raw_dir), "seed": seed, "format_version": 1}
    if output_dir.exists():
        if ((output_dir / "preparation.json").is_file()
                and json.loads((output_dir / "preparation.json").read_text()) == config):
            return output_dir / "dataset.yaml"
        raise ValueError(f"{output_dir} já existe com outra configuração ou está incompleta; use outro --prepared.")

    photos, rejected, parents = {}, [], {}

    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]

    for annotation in sorted(raw_dir.rglob("*.cat")):
        image_path = annotation.with_suffix("")
        try:
            with Image.open(image_path) as image:
                # As coordenadas referem-se aos pixels armazenados, sem rotação EXIF.
                rgb = image.convert("RGB")
                label = read_annotation(annotation, *rgb.size)
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            rejected.append({"annotation": str(annotation), "reason": str(exc)})
            continue
        # Nomes como 00000001_000 e 00000001_005 podem repetir a mesma foto/cabeça.
        photo_id = image_path.stem.rsplit("_", 1)[0]
        find(photo_id)
        if digest not in photos:
            photos[digest] = {"path": image_path, "photo_id": photo_id, "labels": set(), "annotations": []}
        else:
            parents[find(photo_id)] = find(photos[digest]["photo_id"])
        record = photos[digest]
        record["labels"].add(tuple(label))
        record["annotations"].append(str(annotation.relative_to(raw_dir)))

    groups = sorted({find(record["photo_id"]) for record in photos.values()})
    if len(groups) < 10:
        raise ValueError("São necessários pelo menos 10 grupos de fotos válidas para os três splits.")
    random.Random(seed).shuffle(groups)
    train_end, val_end = int(len(groups) * 0.7), int(len(groups) * 0.85)
    splits = {group: "train" if i < train_end else "val" if i < val_end else "test"
              for i, group in enumerate(groups)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "prepared"
        for split in ("train", "val", "test"):
            (staging / "images" / split).mkdir(parents=True)
            (staging / "labels" / split).mkdir(parents=True)
        manifest = []
        for digest, record in photos.items():
            group = find(record["photo_id"])
            split = splits[group]
            # Regrava sem EXIF para que o leitor YOLO não gire as coordenadas.
            with Image.open(record["path"]) as image:
                image.convert("RGB").save(staging / "images" / split / f"{digest}.jpg", quality=95)
            lines = [" ".join(f"{v:.8f}" for v in label) for label in sorted(record["labels"])]
            (staging / "labels" / split / f"{digest}.txt").write_text("\n".join(lines) + "\n")
            manifest.append({"image_sha256": digest, "group": group, "split": split,
                             "source_image": str(record["path"]), "heads": len(lines),
                             "annotations": json.dumps(record["annotations"])})
        metadata = {"path": str(output_dir), "train": "images/train", "val": "images/val",
                    "test": "images/test", "names": {0: "cat_head"}, "kpt_shape": [9, 3],
                    "kpt_names": {0: list(KEYPOINT_NAMES)}}
        (staging / "dataset.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
        pd.DataFrame(manifest).to_csv(staging / "manifest.csv", index=False)
        pd.DataFrame(rejected, columns=["annotation", "reason"]).to_csv(staging / "rejections.csv", index=False)
        (staging / "preparation.json").write_text(json.dumps(config, indent=2))
        staging.rename(output_dir)
    print(f"CAT preparado: {len(photos)} imagens únicas, {len(groups)} grupos, {len(rejected)} anotações rejeitadas.")
    return output_dir / "dataset.yaml"


def train_landmarks(data_yaml, output_path, epochs=30, image_size=640, batch=16,
                    device="cpu", seed=42, initial_model="yolov8n-pose.pt"):
    """Treina nove landmarks, exporta o melhor checkpoint e avalia o teste separado."""
    if epochs < 1 or batch < 1 or image_size < 32:
        raise ValueError("epochs/batch devem ser positivos e image_size >= 32.")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pose = _make_yolo(initial_model)
    pose.train(data=str(data_yaml), epochs=epochs, imgsz=image_size, batch=batch,
               device=str(device), seed=seed, workers=0, patience=10,
               project=str(output_path.parent / "landmark_runs"), name="cat_pose",
               fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0, plots=False)
    # Não usamos flip: a ordem esquerda/direita das orelhas requer uma permutação validada.
    best = Path(pose.trainer.best)
    if not best.is_file():
        raise RuntimeError("O treinamento não produziu best.pt.")
    trained = load_landmark_model(best)
    metrics = trained.val(data=str(data_yaml), split="test", imgsz=image_size,
                          batch=batch, device=str(device), workers=0, plots=False)
    shutil.copy2(best, output_path)
    history_path = output_path.with_suffix(".csv")
    shutil.copy2(Path(pose.trainer.save_dir) / "results.csv", history_path)
    output_path.with_suffix(".json").write_text(json.dumps({
        "source": SOURCE, "keypoints": list(KEYPOINT_NAMES), "seed": seed,
        "data_yaml": str(Path(data_yaml).resolve()), "epochs_requested": epochs,
        "history_csv": str(history_path),
        "test_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "note": "Pose mAP usa OKS uniforme da Ultralytics para 9 pontos; não é acurácia de raça.",
    }, indent=2))
    return output_path


def load_landmark_model(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Modelo de pontos faciais ausente: {path}. Execute train_cat_landmarks.py primeiro.")
    model = _make_yolo(str(path))
    if (model.task != "pose" or list(getattr(model.model.model[-1], "kpt_shape", [])) != [9, 3]
            or model.names != {0: "cat_head"}):
        raise ValueError("Use o modelo CAT treinado com 9 pontos e classe cat_head, não o pose humano.")
    return model


def detect_landmarks(crop, model, device="cpu", conf=0.25, point_conf=0.5):
    """Detecta a cabeça mais confiante no recorte; coordenadas relativas ao recorte."""
    result = model.predict(crop, device=str(device), conf=conf, verbose=False)[0]
    if result.keypoints is None or len(result.boxes) == 0:
        return None
    index = int(result.boxes.conf.argmax().item())
    points = result.keypoints.data[index].cpu().numpy()
    if points.shape != (9, 3):
        raise ValueError("Esperados nove pontos com coordenadas e confiança.")
    width, height = crop.size
    landmarks = {}
    for name, (x, y, confidence) in zip(KEYPOINT_NAMES, points):
        valid = (np.isfinite([x, y, confidence]).all() and confidence >= point_conf
                 and 0 <= x < width and 0 <= y < height)
        landmarks[name] = {"x": float(x), "y": float(y), "confidence": float(confidence), "visible": bool(valid)}
    return {"head_bbox": result.boxes.xyxy[index].cpu().tolist(),
            "confidence": float(result.boxes.conf[index].item()), "points": landmarks}


def draw_landmarks(image, features, offset=(0, 0)):
    """Desenha pontos numerados; funciona com arrays RGB ou BGR (cores simétricas)."""
    annotated = image.copy()
    if features is None:
        return annotated
    for index, point in enumerate(features["points"].values(), start=1):
        if point["visible"]:
            xy = (round(point["x"] + offset[0]), round(point["y"] + offset[1]))
            cv2.circle(annotated, xy, 3, (0, 255, 0), -1)
            cv2.putText(annotated, str(index), (xy[0] + 4, xy[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return annotated


def landmark_table(features):
    """Relatório por ponto; confiança não significa acerto da localização."""
    names = ("Olho esquerdo", "Olho direito", "Boca", "Orelha esquerda 1",
             "Orelha esquerda 2", "Orelha esquerda 3", "Orelha direita 1",
             "Orelha direita 2", "Orelha direita 3")
    rows = []
    for index, (key, name) in enumerate(zip(KEYPOINT_NAMES, names), start=1):
        point = (features or {}).get("points", {}).get(key)
        rows.append({"Ponto": index, "Feature": name,
                     "x (crop)": point["x"] if point and point["visible"] else None,
                     "y (crop)": point["y"] if point and point["visible"] else None,
                     "Confiança": point["confidence"] if point else None,
                     "Status": ("Exibido" if point["visible"] else "Abaixo do limiar/fora do crop")
                     if point else "Sem detecção da cabeça"})
    return pd.DataFrame(rows)


def show_landmark_results(checkpoint, model, prepared_dir, device="cpu", examples=3, conf=0.25, seed=42):
    """Exibe histórico, métricas salvas e exemplos do teste sem retreinar/baixar dados."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    checkpoint, prepared_dir = Path(checkpoint), Path(prepared_dir)
    metadata_path = checkpoint.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metrics = metadata.get("test_metrics", {})
    if metrics:
        print("Métricas do teste do modelo de pontos faciais (salvas no treinamento):")
        display(pd.DataFrame(metrics.items(), columns=["Métrica", "Valor"]))
        print("mAP de pose usa OKS; confiança de um ponto não é acurácia de localização.")
    else:
        print("Métricas de teste não disponíveis junto a este checkpoint.")

    history_path = checkpoint.with_suffix(".csv")
    if not history_path.is_file() and metadata.get("history_csv"):
        history_path = Path(metadata["history_csv"])
    if history_path.is_file():
        history = pd.read_csv(history_path)
        history.columns = history.columns.str.strip()
        groups = [(["train/pose_loss", "val/pose_loss"], "Loss dos pontos faciais"),
                  (["train/box_loss", "val/box_loss"], "Loss da caixa da cabeça"),
                  (["metrics/mAP50(P)", "metrics/mAP50-95(P)"], "mAP de pose — validação")]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, (columns, title) in zip(axes, groups):
            available = [column for column in columns if column in history]
            for column in available:
                ax.plot(history["epoch"], history[column], label=column)
            ax.set(title=title, xlabel="Época")
            if available:
                ax.legend()
            else:
                ax.text(.5, .5, "Histórico indisponível", ha="center", transform=ax.transAxes)
        plt.tight_layout()
        plt.show()
    else:
        print("Histórico CSV indisponível; novos treinamentos o exportam junto ao .pt.")

    manifest_path = prepared_dir / "manifest.csv"
    if not manifest_path.is_file() or examples <= 0:
        print("Exemplos do CAT não exibidos (dataset local ausente ou exemplos desativados).")
        return
    manifest = pd.read_csv(manifest_path)
    test = manifest[manifest["split"] == "test"]
    sample = test.sample(n=min(examples, len(test)), random_state=seed)
    for _, row in sample.iterrows():
        image_path = prepared_dir / "images" / "test" / f"{row['image_sha256']}.jpg"
        label_path = prepared_dir / "labels" / "test" / f"{row['image_sha256']}.txt"
        with Image.open(image_path) as image:
            crop = image.convert("RGB")
        truth = np.array(crop)
        for line in label_path.read_text().splitlines():
            points = np.asarray(line.split()[5:], dtype=float).reshape(9, 3)
            annotations = {"points": {
                name: {"x": float(x * crop.width), "y": float(y * crop.height), "visible": visibility > 0}
                for name, (x, y, visibility) in zip(KEYPOINT_NAMES, points)
            }}
            truth = draw_landmarks(truth, annotations)
        features = detect_landmarks(crop, model, device, conf=conf)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(truth)
        axes[0].set_title("Anotações do dataset — todas as cabeças")
        axes[1].imshow(draw_landmarks(np.array(crop), features))
        count = sum(p["visible"] for p in features["points"].values()) if features else 0
        axes[1].set_title(f"Previsão — cabeça mais confiante ({count}/9 pontos)")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        display(landmark_table(features).round(3))
