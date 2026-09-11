import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from cat_datasets import clean_images, load_cat_breeds, normalize_breed


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


if __name__ == "__main__":
    unittest.main()
