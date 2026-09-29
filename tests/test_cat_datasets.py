import tempfile
import unittest
import contextlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from PIL import Image

from cat_datasets import DATASET_FOLDER, clean_images, load_cat_breeds, normalize_breed


class DatasetTests(unittest.TestCase):
    def test_label_matching(self):
        for label in ("maine_coon_cat", "Maine Coon", "Maine_Coon"):
            self.assertEqual(normalize_breed(label), "Maine_Coon")
        self.assertEqual(normalize_breed("american_shorthair_cat"), "American_Shorthair")
        with self.assertRaises(ValueError):
            normalize_breed("unreviewed_breed_cat")

    def test_duplicates_conflicts_and_invalid_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, (breed, color) in enumerate([
                ("Siamese", "red"), ("Siamese", "red"),
                ("Persian", "blue"), ("Siamese", "blue"),
                ("Manx", "green"),
            ]):
                path = root / f"{i}.png"
                Image.new("RGB", (8, 8), color).save(path)
                rows.append({"filepath": str(path), "breed_name": breed})
            broken = root / "broken.png"
            broken.write_bytes(b"invalid")
            rows.append({"filepath": str(broken), "breed_name": "Manx"})
            cleaned, rejected = clean_images(pd.DataFrame(rows))
            self.assertEqual(cleaned.breed_name.tolist(), ["Siamese", "Manx"])
            self.assertEqual(cleaned.filepath.iloc[0], rows[0]["filepath"])
            self.assertEqual(rejected.reason.value_counts().to_dict(), {
                "conflicting_labels": 2, "invalid_image": 1, "duplicate_image": 1,
            })

    def test_folder_labels_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / "devon_rex_cat"
            folder.mkdir()
            Image.new("RGB", (8, 8)).save(folder / "000.png")
            frame = load_cat_breeds(root)
            self.assertEqual(frame.breed_name.tolist(), ["Devon_Rex"])
            self.assertEqual(frame.original_label.tolist(), ["devon_rex_cat"])
            self.assertEqual(frame.source.tolist(), ["atharva_taras"])

    def test_notebook_dataset_modes(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               "cnn_gato_deteccao_classificacao.ipynb").read_text())
        code_cells = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        cells = [next(source for source in code_cells if 'DATASET_MODE = CONFIG[' in source),
                 next(source for source in code_cells if 'dataset_frames = []' in source)]
        for mode in ("oxford", "cat_breeds", "both"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                # A fonte não selecionada não existe: qualquer leitura indevida falha.
                if mode in ("oxford", "both"):
                    oxford = root / "data" / "oxford_pets"
                    (oxford / "images").mkdir(parents=True)
                    (oxford / "annotations").mkdir()
                    (oxford / "annotations" / "list.txt").write_text("Siamese_1 1 1 1\n")
                    Image.new("RGB", (8, 8), "red").save(oxford / "images" / "Siamese_1.jpg")
                if mode in ("cat_breeds", "both"):
                    folder = root / "data" / DATASET_FOLDER / "images" / "devon_rex_cat"
                    folder.mkdir(parents=True)
                    Image.new("RGB", (8, 8), "blue").save(folder / "000.png")
                namespace = {"CONFIG": {"DATASET_MODE": mode}, "Path": Path, "os": os, "pd": pd}
                with patch.object(Path, "cwd", return_value=root), \
                     patch("cat_datasets.urlopen", side_effect=AssertionError("Download indevido")), \
                     contextlib.redirect_stdout(io.StringIO()):
                    for cell in cells:
                        exec(cell, namespace)
                    expected = ({"oxford_iiit_pet"} if mode == "oxford" else
                                {"atharva_taras"} if mode == "cat_breeds" else
                                {"oxford_iiit_pet", "atharva_taras"})
                    self.assertEqual(set(namespace['cats_df'].source), expected)
                    # Reexecutar não deve acumular imagens da execução anterior.
                    exec(cells[1], namespace)
                    self.assertEqual(len(namespace['cats_df']), len(expected))
        with self.assertRaisesRegex(ValueError, "DATASET_MODE"):
            exec(cells[0], {"CONFIG": {"DATASET_MODE": "typo"}, "Path": Path})


if __name__ == "__main__":
    unittest.main()
