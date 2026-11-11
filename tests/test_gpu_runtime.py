import unittest
from unittest.mock import patch
from yut_ai.runtime import DEFAULT_DEVICE, DEFAULT_WORKERS, DEFAULT_INFERENCE_BACKEND, validate_inference_device, device_for

class GPURuntimeTests(unittest.TestCase):
    def test_defaults_and_worker_guard(self):
        self.assertEqual((DEFAULT_DEVICE,DEFAULT_WORKERS,DEFAULT_INFERENCE_BACKEND),('cpu',4,'torch'))
        validate_inference_device('cpu',4,'torch')
        validate_inference_device('cpu',4,'torch')
        with self.assertRaises(ValueError):validate_inference_device('mps',4,'torch')

    def test_gpu_request_does_not_silently_fall_back(self):
        with patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaises(RuntimeError):device_for('mps')
            self.assertEqual(str(device_for('cpu')),'cpu')

    def test_single_worker_preserves_gpu_model(self):
        from yut_ai.parallel import GameWorkers
        model=object()
        with patch('yut_ai.selfplay.play',return_value=('game',[],[])) as play:
            workers=GameWorkers(1,'torch')
            workers.generate(model,[(42,4,None,20,8)])
            self.assertIs(play.call_args.args[0],model)
