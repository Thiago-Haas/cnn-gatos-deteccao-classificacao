"""Exporta YOLOv8n e o checkpoint ResNet50 e verifica paridade numérica."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cat_calibration import load_calibration, DEFAULT_CALIBRATION

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn
from torchvision import models
from ultralytics import YOLO


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", type=Path, default=project / "artifacts/cat_breed_classifier.pt")
    parser.add_argument("--detector", type=Path, default=project / "yolov8n.pt")
    parser.add_argument("--output", type=Path, default=project / "artifacts/onnx")
    parser.add_argument("--detector-size", type=int, default=320)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--no-calibration", action="store_true")
    args = parser.parse_args()
    if args.detector_size < 32 or args.detector_size % 32:
        parser.error("--detector-size deve ser múltiplo de 32.")
    if not args.classifier.is_file() or not args.detector.is_file():
        parser.error("Informe os dois checkpoints locais existentes.")
    calibration = load_calibration(args.classifier, args.calibration, enabled=not args.no_calibration)
    torch.set_num_threads(2)
    checkpoint = torch.load(args.classifier, map_location="cpu", weights_only=True)
    labels = {int(k): v for k, v in checkpoint["idx_to_breed"].items()}
    if not labels or sorted(labels) != list(range(len(labels))):
        raise ValueError("Mapeamento de raças inválido")
    size = int(checkpoint.get("config", {}).get("IMG_SIZE", 224))
    model = models.resnet50(weights=None)
    state = checkpoint["model_state_dict"]
    linear = nn.Linear(model.fc.in_features, len(labels))
    model.fc = nn.Sequential(nn.Dropout(float(checkpoint.get("config", {}).get("DROPOUT", 0.4))), linear) if "fc.1.weight" in state else linear
    model.load_state_dict(state, strict=True)
    model.eval()
    args.output.mkdir(parents=True, exist_ok=True)
    # Só publica os arquivos após ambas as exportações e verificações passarem.
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        sample = torch.randn(1, 3, size, size, generator=torch.Generator().manual_seed(42))
        classifier_path = staging / "classifier.onnx"
        torch.onnx.export(model, sample, str(classifier_path), opset_version=17,
                          input_names=["images"], output_names=["logits"], dynamo=False)
        onnx.checker.check_model(str(classifier_path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(classifier_path), options, providers=["CPUExecutionProvider"])
        actual = session.run(None, {"images": sample.numpy()})[0]
        with torch.inference_mode():
            expected = model(sample).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        report = {"classifier_max_abs_error": float(np.max(np.abs(actual - expected)))}
        shutil.copy2(args.detector, staging / "detector.pt")
        yolo = YOLO(str(staging / "detector.pt"))
        if yolo.task != "detect" or yolo.names.get(15) != "cat" or len(yolo.names) != 80:
            raise ValueError("Use YOLOv8 detector COCO com 80 classes e gato no índice 15.")
        exported = Path(yolo.export(format="onnx", imgsz=args.detector_size, opset=17,
                                  simplify=False, dynamic=False, nms=False, device="cpu"))
        onnx.checker.check_model(str(exported))
        session = ort.InferenceSession(str(exported), options, providers=["CPUExecutionProvider"])
        detector_input = np.random.default_rng(42).random((1, 3, args.detector_size, args.detector_size), dtype=np.float32)
        actual = session.run(None, {session.get_inputs()[0].name: detector_input})[0]
        with torch.inference_mode():
            expected = yolo.model(torch.from_numpy(detector_input))
            if isinstance(expected, (tuple, list)):
                expected = expected[0]
            expected = expected.numpy()
        if actual.ndim != 3 or actual.shape[1] != 84:
            raise ValueError(f"Saída YOLO incompatível: {actual.shape}; esperado [1,84,N].")
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
        report["detector_max_abs_error"] = float(np.max(np.abs(actual - expected)))
        def digest(path):
            with path.open('rb') as file:
                value = hashlib.sha256()
                for chunk in iter(lambda: file.read(1024 * 1024), b''):
                    value.update(chunk)
                return value.hexdigest()
        metadata = {"version": 2, "classifierCalibration": calibration,
                    "sourceRepository": "https://github.com/Thiago-Haas/cnn-gatos-deteccao-classificacao",
                    "training": {"experiment": checkpoint.get("experimento"),
                                 "classifierCheckpointSha256": digest(args.classifier),
                                 "datasetMode": checkpoint.get("config", {}).get("DATASET_MODE")},
                    "modelHashes": {"detector": digest(exported), "classifier": digest(classifier_path)}, "labels": [labels[i] for i in range(len(labels))],
                    "classifierSize": size, "detectorSize": args.detector_size,
                    "catClass": 15, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
                    "validation": report,
                    "assetBytes": {"detector": exported.stat().st_size, "classifier": classifier_path.stat().st_size}}
        shutil.copy2(classifier_path, args.output / "classifier.onnx")
        shutil.copy2(exported, args.output / "detector.onnx")
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("Pacote web salvo em", args.output)
    print("Copie classifier.onnx, detector.onnx e metadata.json juntos para mobile-onnx/models/.")


if __name__ == "__main__":
    main()
