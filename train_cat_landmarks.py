"""Baixa/prepara o CAT Dataset e treina o detector de nove pontos faciais."""

import argparse
from pathlib import Path

import torch

from cat_landmarks import download_landmarks, prepare_landmarks, train_landmarks


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, help="Pasta já extraída com imagens e .cat; dispensa download.")
    parser.add_argument("--archive", type=Path, help="ZIP já baixado do Kaggle.")
    parser.add_argument("--prepared", type=Path, default=root / "data" / "cat_landmarks_pose")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "cat_landmarks.pt")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.raw and args.archive:
        parser.error("Use somente --raw ou --archive.")
    if args.epochs < 1 or args.batch < 1 or args.image_size < 32:
        parser.error("--epochs/--batch devem ser positivos; --image-size deve ser >= 32.")
    raw = args.raw or download_landmarks(root / "data", args.archive)
    yaml_path = prepare_landmarks(raw, args.prepared, seed=args.seed)
    print("Dataset:", yaml_path)
    if not args.prepare_only:
        checkpoint = train_landmarks(yaml_path, args.output, epochs=args.epochs,
                                     image_size=args.image_size, batch=args.batch,
                                     device=args.device, seed=args.seed)
        print("Modelo de pontos faciais salvo em", checkpoint)


if __name__ == "__main__":
    main()
