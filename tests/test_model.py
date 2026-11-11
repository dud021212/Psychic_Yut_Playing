import tempfile
import unittest
from pathlib import Path
import torch
from yut_ai.model import ModelConfig, PolicyValueNet
from yut_ai.data import Example, Replay, collate
from yut_ai.training import Trainer
from yut_ai.runtime import cutoff

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        self.config = ModelConfig(8, 4, 32)
        self.examples = [Example(torch.randn(8), torch.randn(n, 4),
                                 torch.ones(n) / n, 0.5) for n in (1, 3)]

    def test_padding_and_action_permutation(self):
        model = PolicyValueNet(self.config)
        s, a, mask, _, _ = collate(self.examples, 'cpu')
        probabilities, _ = model.predict(s, a, mask)
        self.assertEqual(float(probabilities[0, 1:].sum()), 0)
        permutation = torch.tensor([2, 0, 1])
        changed, _ = model.predict(s[1:], a[1:, permutation], mask[1:, permutation])
        torch.testing.assert_close(changed, probabilities[1:, permutation])
        with self.assertRaises(ValueError):
            model.predict(s, a, torch.zeros_like(mask))

    def test_training_and_resume(self):
        trainer = Trainer(PolicyValueNet(self.config), schema='test')
        first = trainer.train_batch(self.examples)['loss']
        for _ in range(40):
            last = trainer.train_batch(self.examples)['loss']
        self.assertLess(last, first)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            trainer.save(path)
            restored = Trainer.load(path, schema='test')
            trainer.train_batch(self.examples)
            restored.train_batch(self.examples)
            for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
                torch.testing.assert_close(a, b)
            with self.assertRaises(ValueError):
                Trainer.load(path, schema='wrong')

    def test_replay_and_depth(self):
        replay = Replay(self.config, max_bytes=120)
        replay.add(self.examples[0])
        replay.add(self.examples[1])
        self.assertLessEqual(replay.bytes, 120)
        self.assertEqual(len(replay.items), 1)
        self.assertEqual(len(replay.sample(4)), 4)
        self.assertFalse(cutoff('easy', 3))
        self.assertTrue(cutoff('easy', 4))
        self.assertTrue(cutoff('hard', 16))
        self.assertFalse(cutoff('expert', 1000))

if __name__ == '__main__':
    unittest.main()
