import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch
import yaml

from cat_landmarks import (
    KEYPOINT_NAMES, detect_landmarks, draw_landmarks, load_landmark_model,
    prepare_landmarks, read_annotation, landmark_table, show_landmark_results,
)


ANNOTATION = "9 20 25 40 25 30 40 10 20 15 5 22 18 38 18 45 5 50 20"


def make_fixture(root, count=12):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        image = root / f"{i:08d}_000.jpg"
        Image.new("RGB", (64, 64), (i * 15 % 255, 80, 120)).save(image)
        Path(str(image) + ".cat").write_text(ANNOTATION)


class LandmarkTests(unittest.TestCase):
    def test_annotation_normalization_and_missing_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg.cat"
            path.write_text(ANNOTATION)
            label = read_annotation(path, 64, 64)
            self.assertEqual(label.shape, (32,))
            points = label[5:].reshape(9, 3)
            np.testing.assert_allclose(points[0], [20 / 64, 25 / 64, 2])
            self.assertTrue(((label[1:5] >= 0) & (label[1:5] <= 1)).all())
            path.write_text(ANNOTATION.replace("20 25", "-1 -1", 1))
            np.testing.assert_array_equal(read_annotation(path, 64, 64)[5:8], [0, 0, 0])
            path.write_text("9 1 2")
            with self.assertRaises(ValueError):
                read_annotation(path, 64, 64)

    def test_grouped_splits_merge_annotations_and_reject_bad_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            make_fixture(raw)
            # Mesmos pixels, outra cabeça anotada: preservar as duas linhas de pose.
            duplicate = raw / "00000000_001.jpg"
            duplicate.write_bytes((raw / "00000000_000.jpg").read_bytes())
            Path(str(duplicate) + ".cat").write_text(ANNOTATION.replace("20 25", "21 25", 1))
            # Mesma família de foto com pixels diferentes: deve ficar no mesmo split.
            variant = raw / "00000000_002.jpg"
            Image.new("RGB", (64, 64), "green").save(variant)
            Path(str(variant) + ".cat").write_text(ANNOTATION)
            (raw / "bad.jpg.cat").write_text(ANNOTATION)
            prepared = root / "prepared"
            config_path = prepare_landmarks(raw, prepared)
            manifest = pd.read_csv(prepared / "manifest.csv", dtype={"group": str})
            self.assertEqual(len(manifest), 13)
            self.assertEqual(manifest.heads.max(), 2)
            self.assertEqual(manifest.groupby('group').split.nunique().max(), 1)
            self.assertEqual(set(manifest.split), {"train", "val", "test"})
            self.assertEqual(len(pd.read_csv(prepared / "rejections.csv")), 1)
            self.assertEqual(yaml.safe_load(config_path.read_text())["kpt_shape"], [9, 3])
            self.assertEqual(prepare_landmarks(raw, prepared), config_path)
            with self.assertRaises(ValueError):
                prepare_landmarks(raw, prepared, seed=7)
            second = root / "second"
            prepare_landmarks(raw, second)
            pd.testing.assert_frame_equal(pd.read_csv(prepared / "manifest.csv"),
                                          pd.read_csv(second / "manifest.csv"))

    def test_inference_no_detection_and_confidence_filter(self):
        crop = Image.new("RGB", (64, 64))
        points = torch.tensor([[[20., 25., .9]] * 9])
        points[0, 1, 2] = .1
        result = SimpleNamespace(
            boxes=SimpleNamespace(conf=torch.tensor([.8]), xyxy=torch.tensor([[1., 2., 50., 55.]])),
            keypoints=SimpleNamespace(data=points),
        )
        # Boxes precisa suportar len, como Results da Ultralytics.
        class Boxes:
            conf = result.boxes.conf
            xyxy = result.boxes.xyxy

            def __len__(self):
                return 1

        result.boxes = Boxes()
        model = SimpleNamespace(predict=lambda *a, **k: [result])
        features = detect_landmarks(crop, model)
        self.assertTrue(features['points']['left_eye']['visible'])
        self.assertFalse(features['points']['right_eye']['visible'])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        drawn = draw_landmarks(frame, features, offset=(10, 10))
        self.assertTrue(drawn[35, 30].any())
        self.assertFalse(frame.any())
        result.keypoints = None
        self.assertIsNone(detect_landmarks(crop, model))

    def test_human_pose_checkpoint_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as checkpoint:
            human = SimpleNamespace(task="pose", names={0: "person"},
                                    model=SimpleNamespace(model=[SimpleNamespace(kpt_shape=[17, 3])]))
            with patch("cat_landmarks._make_yolo", return_value=human):
                with self.assertRaisesRegex(ValueError, "modelo CAT"):
                    load_landmark_model(checkpoint.name)

    def test_notebook_off_does_not_download_or_train(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'cnn_gato_deteccao_classificacao.ipynb').read_text())
        source = next(''.join(c['source']) for c in notebook['cells']
                      if 'landmark_mode = CONFIG[' in ''.join(c['source']))
        with patch('cat_landmarks.download_landmarks', side_effect=AssertionError('download')):
            namespace = {'CONFIG': {'LANDMARKS_MODE': 'off'}, 'Path': Path}
            exec(source, namespace)
            self.assertIsNone(namespace['landmark_model'])

    def test_optional_results_and_pipeline_panels(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        notebook = json.loads((Path(__file__).resolve().parents[1] /
                               'cnn_gato_deteccao_classificacao.ipynb').read_text())
        results_cell = next(''.join(c['source']) for c in notebook['cells']
                            if 'show_landmark_results(' in ''.join(c['source']))
        with patch('cat_landmarks.show_landmark_results', side_effect=AssertionError('Resultados desativados')):
            exec(results_cell, {'CONFIG': {'LANDMARKS_MODE': 'off'}})
            exec(results_cell, {'CONFIG': {'LANDMARKS_MODE': 'load', 'LANDMARKS_SHOW_RESULTS': False}})
        pipeline_cell = next(''.join(c['source']) for c in notebook['cells']
                             if 'def full_pipeline(' in ''.join(c['source']))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cat.jpg'
            Image.new('RGB', (64, 64)).save(path)
            namespace = {'Image': Image, 'np': np, 'plt': plt, 'device': 'cpu',
                         'landmark_model': object(), 'draw_landmarks': draw_landmarks,
                         'detect_cat_bbox': lambda _: ([0, 0, 64, 64], .9),
                         'generate_gradcam': lambda _: (np.zeros((64, 64, 3)), 'Siamese', .8)}
            exec(pipeline_cell, namespace)
            for mode, expected_panels in [('off', 3), ('load', 4)]:
                namespace['CONFIG'] = {'LANDMARKS_MODE': mode, 'LANDMARKS_CONF': .25}
                tables, panels, detections = [], [], []
                namespace['display'] = tables.append
                namespace['detect_landmarks'] = lambda *a, **k: detections.append(True)
                with patch.object(plt, 'show', side_effect=lambda: panels.append(len(plt.gcf().axes))):
                    namespace['full_pipeline'](path)
                plt.close('all')
                self.assertEqual(panels, [expected_panels])
                self.assertEqual(len(tables), int(mode == 'load'))
                self.assertEqual(len(detections), int(mode == 'load'))

    def test_results_show_history_metrics_and_annotated_examples(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_fixture(root / 'raw')
            prepare_landmarks(root / 'raw', root / 'prepared')
            checkpoint = root / 'cat_landmarks.pt'
            checkpoint.with_suffix('.json').write_text(json.dumps({'test_metrics': {'metrics/mAP50(P)': .4}}))
            pd.DataFrame({'epoch': [1, 2], 'train/pose_loss': [3, 2], 'val/pose_loss': [4, 3],
                          'metrics/mAP50(P)': [.2, .3]}).to_csv(checkpoint.with_suffix('.csv'), index=False)
            panels, tables = [], []
            with patch('cat_landmarks.detect_landmarks', return_value=None), \
                 patch('IPython.display.display', side_effect=tables.append), \
                 patch.object(plt, 'show', side_effect=lambda: panels.append(len(plt.gcf().axes))):
                show_landmark_results(checkpoint, object(), root / 'prepared', examples=2)
            plt.close('all')
            self.assertEqual(panels, [3, 2, 2])
            self.assertEqual(len(tables), 3)
            self.assertEqual(len(tables[-1]), 9)
            self.assertTrue(tables[-1]['Confiança'].isna().all())
            self.assertTrue(landmark_table(None)['x (crop)'].isna().all())

    def test_webcam_runs_landmarks_before_breed_and_keeps_rgb_crop(self):
        import webcam

        events = []
        box = SimpleNamespace(xyxy=torch.tensor([[10., 10., 40., 40.]]))
        detector = SimpleNamespace(predict=lambda **kwargs: [SimpleNamespace(boxes=[box])])

        def landmarks(crop, model, device, conf):
            events.append('landmarks')
            self.assertEqual(crop.size, (30, 30))
            return None  # Sem cabeça confiável, a raça ainda deve ser calculada.

        def transform(crop):
            self.assertTrue((np.asarray(crop) == 50).all())
            return torch.zeros(3, 8, 8)

        def classifier(tensor):
            events.append('breed')
            return torch.tensor([[1.]])

        frame = np.full((64, 64, 3), 50, dtype=np.uint8)
        with patch('webcam.detect_landmarks', side_effect=landmarks):
            output = webcam.annotate_frame(frame, detector, classifier, {0: 'Siamese'},
                                            transform, torch.device('cpu'), .25,
                                            landmark_model=object())
        self.assertEqual(events, ['landmarks', 'breed'])
        self.assertEqual(output.shape, frame.shape)
        self.assertTrue((frame == 50).all())


if __name__ == '__main__':
    unittest.main()
