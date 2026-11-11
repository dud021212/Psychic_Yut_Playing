import copy
import random
import unittest
import torch
from yut_ai.game import Game, paths
from yut_ai.rules import MoveResult
from yut_ai.player import STATE_DIM, ACTION_DIM
from yut_ai.model import ModelConfig, PolicyValueNet
from yut_ai.parallel import GameWorkers

class ParallelTests(unittest.TestCase):
    def test_clone_independent_and_equivalent(self):
        game = Game.new(random.Random(4))
        game.moves = [MoveResult('do', 1)]
        game.stored[0] = [MoveResult('mo', 5)]
        clone = game.clone()
        self.assertEqual(clone.__dict__.keys(), game.__dict__.keys())
        self.assertEqual(clone.legal_actions(), copy.deepcopy(game).legal_actions())
        clone.hands[0][0] = 99
        clone.positions[0][0] = 2
        clone.used[0][0] = True
        clone.moves.clear(); clone.stored[0].clear()
        clone.installations[2] = 'gate'
        clone.beliefs[0].probabilities[0] = 99
        self.assertNotEqual(clone.hands, game.hands)
        self.assertEqual(game.positions[0][0], -1)
        self.assertFalse(game.used[0][0])
        self.assertEqual(len(game.moves), 1)
        self.assertEqual(len(game.stored[0]), 1)
        self.assertFalse(game.installations)
        self.assertNotEqual(game.beliefs[0].probabilities[0], 99)
        route = paths(5, MoveResult('mo', 5)); route.clear()
        self.assertTrue(paths(5, MoveResult('mo', 5)))

    def test_parallel_same_frozen_games(self):
        torch.set_num_threads(1); torch.manual_seed(5)
        model = PolicyValueNet(ModelConfig(STATE_DIM, ACTION_DIM))
        arguments = [(seed, 4, None, 1000, 8) for seed in (42, 43)]
        serial, parallel = GameWorkers(1, 'torch'), GameWorkers(2, 'torch')
        try:
            a = serial.generate(model, arguments)
            b = parallel.generate(model, arguments)
            for ((ga, ea, events_a), _), ((gb, eb, events_b), _) in zip(a, b):
                self.assertEqual(events_a, events_b)
                self.assertEqual(ga.winner, gb.winner)
                self.assertEqual(len(ea), len(eb))
                for x, y in zip(ea, eb):
                    torch.testing.assert_close(x.state, y.state)
                    torch.testing.assert_close(x.policy, y.policy)
        finally:
            serial.close(); parallel.close()

if __name__ == '__main__': unittest.main()
