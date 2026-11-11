"""Optional Apple CPU + Neural Engine inference; training stays in PyTorch.

Fixed action chunks avoid flexible shapes and never discard legal actions.
Core ML decides placement: CPU_AND_NE permits ANE, it does not prove ANE use.
"""
import copy
import platform
from pathlib import Path
import numpy as np
import torch

ACTION_CHUNK = 32


def require_coreml():
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('coreml-ne requires an Apple Silicon Mac with macOS 13+.')
    if int(platform.mac_ver()[0].split('.')[0]) < 13:
        raise RuntimeError('coreml-ne requires macOS 13+.')
    try:
        import coremltools as ct
    except ImportError as exc:
        raise RuntimeError('Core ML dependency missing. See NPU.md for the separate .venv-coreml setup.') from exc
    return ct


class ExportNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, states, actions):
        m = self.model
        h = m.encoder(states)
        joint = torch.relu(m.state_policy(h).unsqueeze(1) + m.action_policy(actions))
        return m.score(joint).squeeze(-1), m.value(h).squeeze(-1)


def export_model(model, path):
    ct = require_coreml()
    model = copy.deepcopy(model).cpu().eval()
    config = model.config
    inputs = (torch.zeros(1, config.state_dim), torch.zeros(1, ACTION_CHUNK, config.action_dim))
    traced = torch.jit.trace(ExportNetwork(model).eval(), inputs)
    converted = ct.convert(
        traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS13,
        inputs=[ct.TensorType(name='states', shape=inputs[0].shape, dtype=np.float32),
                ct.TensorType(name='actions', shape=inputs[1].shape, dtype=np.float32)],
        outputs=[ct.TensorType(name='logits', dtype=np.float32),
                 ct.TensorType(name='value', dtype=np.float32)],
        compute_precision=ct.precision.FLOAT16, skip_model_load=True,
    )
    converted.user_defined_metadata['action_chunk'] = str(ACTION_CHUNK)
    converted.user_defined_metadata['state_dim'] = str(config.state_dim)
    converted.user_defined_metadata['action_dim'] = str(config.action_dim)
    converted.save(str(path))
    return str(path)


class CoreMLPredictor:
    def __init__(self, path):
        ct = require_coreml()
        self.model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        metadata = self.model.user_defined_metadata
        self.chunk = int(metadata['action_chunk'])
        self.state_dim = int(metadata['state_dim'])
        self.action_dim = int(metadata['action_dim'])

    def evaluate_features(self, state, actions):
        states = np.asarray(state, dtype=np.float32).reshape(1, self.state_dim)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or not len(actions):
            raise ValueError('Expected nonempty legal action features')
        scores = []
        value = None
        for start in range(0, len(actions), self.chunk):
            rows = actions[start:start+self.chunk]
            padded = np.zeros((1, self.chunk, self.action_dim), dtype=np.float32)
            padded[0, :len(rows)] = rows
            result = self.model.predict({'states': states, 'actions': padded})
            scores.extend(np.asarray(result['logits']).reshape(-1)[:len(rows)])
            if value is None:
                value = float(np.asarray(result['value']).reshape(-1)[0])
        logits = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(logits).all() or not np.isfinite(value):
            raise FloatingPointError('Nonfinite Core ML prediction')
        probabilities = np.exp(logits-logits.max())
        probabilities /= probabilities.sum()
        return probabilities.tolist(), value


def main():
    import argparse
    import json
    import random
    import tempfile
    from .training import Trainer
    from .player import SCHEMA, evaluate
    from .game import Game
    parser = argparse.ArgumentParser(description='Convert and verify Apple CPU + NE inference')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    require_coreml()
    model = Trainer.load(args.checkpoint, 'cpu', schema=SCHEMA).model.eval()
    with tempfile.TemporaryDirectory(prefix='yut-coreml-check-') as directory:
        path = args.output or Path(directory)/'model.mlpackage'
        if path.exists():
            parser.error('Output already exists; choose a new path')
        export_model(model, path)
        predictor = CoreMLPredictor(path)
        max_error = 0.0
        rng = random.Random(42)
        game = Game.new(rng)
        for _ in range(64):
            actions = game.legal_actions()
            expected = evaluate(model, game, actions)
            actual = evaluate(predictor, game, actions)
            error = max(max(abs(a-b) for a,b in zip(expected[0], actual[0])), abs(expected[1]-actual[1]))
            max_error = max(max_error, error)
            if error > 0.02:
                raise RuntimeError(f'Core ML parity error {error} exceeds FP16 tolerance 0.02')
            game.apply(rng.choice(actions), rng)
            if game.winner is not None:
                game = Game.new(rng)
        print(json.dumps({'prediction_parity': 'passed', 'max_abs_error': max_error,
                          'compute_units': 'CPU_AND_NE', 'ane_execution_verified': False,
                          'output': str(args.output) if args.output else None}))

if __name__ == '__main__':
    main()
