"""Integração do Cat Breeds Dataset com os rótulos do Oxford-IIIT Pet.

Citation: Atharva Taras, 2026. Cat Breeds Dataset.
https://github.com/AtharvaTaras/Cat-Breeds-Dataset (CC BY 4.0).
"""

import hashlib
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlopen
import zipfile

import pandas as pd
from PIL import Image, ImageOps


DATASET_REVISION = "56a69059119626ed210023bbfc6fdcb8262e8703"
DATASET_URL = (
    "https://codeload.github.com/AtharvaTaras/Cat-Breeds-Dataset/zip/"
    + DATASET_REVISION
)
DATASET_FOLDER = "Cat-Breeds-Dataset-" + DATASET_REVISION
OXFORD_BREEDS = (
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx",
)
ADDITIONAL_BREEDS = (
    "American_Shorthair", "Cornish_Rex", "Devon_Rex", "Himalayan", "Manx",
    "Norwegian_Forest", "Oriental_Shorthair", "Savannah", "Scottish_Fold",
    "Turkish_Angora",
)
BREED_LABELS = {name.lower(): name for name in OXFORD_BREEDS + ADDITIONAL_BREEDS}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_breed(name):
    """Unifica caixa/separadores e remove o sufixo _cat usado pelo dataset."""
    key = re.sub(r"[\s_-]+", "_", name.strip().lower()).removesuffix("_cat")
    if key not in BREED_LABELS:
        raise ValueError(f"Raça não mapeada: {name!r}. Revise o mapeamento explicitamente.")
    return BREED_LABELS[key]


def download_cat_breeds(data_dir, archive_path=None):
    """Baixa a revisão fixada; também aceita seu ZIP já baixado para uso offline."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    destination = data_dir / DATASET_FOLDER
    if destination.is_dir():
        if not (destination / "images").is_dir():
            raise ValueError(f"Dataset incompleto: {destination}")
        return destination / "images"
    # Só publica a pasta definitiva depois que toda a extração termina.
    with tempfile.TemporaryDirectory(dir=data_dir) as temporary:
        staging = Path(temporary)
        if archive_path is None:
            archive_path = staging / "dataset.zip"
            print("Baixando Cat Breeds Dataset (~303 MB descompactados)...")
            with urlopen(DATASET_URL, timeout=60) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or ".." in path.parts or path.parts[0] != DATASET_FOLDER:
                    raise ValueError(f"Caminho inesperado no ZIP: {member.filename}")
            archive.extractall(staging)
        if not (staging / DATASET_FOLDER / "images").is_dir():
            raise ValueError("O ZIP não contém a pasta images esperada.")
        (staging / DATASET_FOLDER).rename(destination)
    return destination / "images"


def load_cat_breeds(images_dir):
    """Lê imagens/<raça>_cat/*; mantém a origem e o rótulo original para auditoria."""
    images_dir = Path(images_dir)
    records = []
    for path in sorted(images_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            original = path.relative_to(images_dir).parts[0]
            records.append({
                "filename": path.name,
                "filepath": str(path.resolve()),
                "breed_name": normalize_breed(original),
                "original_label": original,
                "source": "atharva_taras",
            })
    if not records:
        raise ValueError(f"Nenhuma imagem encontrada em {images_dir}")
    return pd.DataFrame(records)


def clean_images(dataframe):
    """Remove imagens ilegíveis, duplicatas RGB exatas e grupos com rótulos conflitantes.

    Executar ANTES do split. Não detecta fotos semelhantes, recortadas ou recomprimidas.
    Em duplicatas com a mesma raça, preserva a primeira origem (Oxford no notebook).
    """
    valid, rejected = [], []
    for record in dataframe.to_dict("records"):
        try:
            with Image.open(record["filepath"]) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            rejected.append({**record, "reason": "invalid_image", "detail": str(exc)})
            continue
        valid.append({**record, "image_sha256": digest})
    if not valid:
        raise ValueError("Nenhuma imagem válida após a leitura do dataset.")
    valid_df = pd.DataFrame(valid)
    conflicts = valid_df.groupby("image_sha256")["breed_name"].transform("nunique") > 1
    for record in valid_df[conflicts].to_dict("records"):
        rejected.append({**record, "reason": "conflicting_labels"})
    valid_df = valid_df[~conflicts]
    duplicates = valid_df.duplicated("image_sha256", keep="first")
    for record in valid_df[duplicates].to_dict("records"):
        rejected.append({**record, "reason": "duplicate_image"})
    cleaned = valid_df[~duplicates].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("Nenhuma imagem restante após remover conflitos e duplicatas.")
    return cleaned, pd.DataFrame(rejected, columns=[
        *dataframe.columns, "image_sha256", "reason", "detail",
    ])
