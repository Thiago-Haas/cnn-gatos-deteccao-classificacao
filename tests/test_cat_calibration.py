import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from cat_calibration import load_calibration
from webcam import annotate_frame


class CalibrationTests(unittest.TestCase):
    def test_checkpoint_binding_and_invalid_parameters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model.pt'; checkpoint.write_bytes(b'fixture')
            config = root / 'srd.json'
            data = {'version':1, 'temperature':0.8706899881362915, 'threshold':0.65,
                    'classifierCheckpointSha256':hashlib.sha256(b'fixture').hexdigest()}
            config.write_text(json.dumps(data))
            self.assertEqual(load_calibration(checkpoint, config)['threshold'], .65)
            for value in (0, -1, float('nan'), float('inf'), True):
                config.write_text(json.dumps({**data, 'temperature':value}))
                with self.assertRaises(ValueError): load_calibration(checkpoint, config)
            config.write_text(json.dumps(data))
            checkpoint.write_bytes(b'new model')
            with self.assertRaisesRegex(ValueError, 'outro checkpoint'): load_calibration(checkpoint, config)
            self.assertEqual(load_calibration(checkpoint, enabled=False), {'temperature':1., 'threshold':.4})

    def test_webcam_applies_temperature_before_rejection(self):
        detector = SimpleNamespace(predict=lambda **kwargs:[SimpleNamespace(boxes=[SimpleNamespace(xyxy=torch.tensor([[0,0,10,10]]))])])
        classifier = lambda x: torch.tensor([[1.2, 0., 0.]])
        frame = np.zeros((10,10,3), dtype=np.uint8)
        transform = lambda crop: torch.zeros(3, 10, 10)
        for temperature, expected in ((1., 'Sem Raça Definida'), (.8706899881362915, 'A ')):
            with patch('webcam.cv2.putText') as put:
                annotate_frame(frame.copy(), detector, classifier, {0:'A',1:'B',2:'C'}, transform,
                               'cpu', .25, breed_conf=.65, temperature=temperature)
                self.assertTrue(put.call_args.args[1].startswith(expected), put.call_args)
