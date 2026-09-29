"""Parâmetros SRD vinculados ao checkpoint que originou o estudo."""
import hashlib
import json
import math
from pathlib import Path

DEFAULT_CALIBRATION = Path(__file__).resolve().parent / "config/srd.json"


def load_calibration(checkpoint, calibration_path=DEFAULT_CALIBRATION, *, enabled=True):
    if not enabled:
        return {"temperature": 1.0, "threshold": 0.40}
    config = json.loads(Path(calibration_path).read_text())
    for key in ("temperature", "threshold"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Calibração: {key} precisa ser um número finito positivo.")
    if config['threshold'] > 1 or config.get('version') != 1:
        raise ValueError("Versão ou limiar de calibração inválido.")
    with Path(checkpoint).open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    if digest != config.get('classifierCheckpointSha256'):
        raise ValueError("Calibração pertence a outro checkpoint. Gere parâmetros para este modelo ou use --no-calibration.")
    return config
