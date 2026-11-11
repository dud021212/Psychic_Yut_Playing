import random
import unittest
from unittest.mock import patch
from yut_ai.game import Game
from yut_ai.player import neural_rollout_action, search

class NeuralRolloutTests(unittest.TestCase):
    def test_each_players_policy_selects_action(self):
        game = Game.new(random.Random(7))
        model = object()
        for player in (0, 1):
            game.player = player
            actions = game.legal_actions()
            with patch('yut_ai.player.evaluate', return_value=([0.0]*(len(actions)-1)+[1.0], 0)) as infer:
                self.assertEqual(neural_rollout_action(game, model, random.Random(1)), actions[-1])
                infer.assert_called_once_with(model, game, actions)
        game.phase = 'throw'
        with patch('yut_ai.player.evaluate') as infer:
            self.assertEqual(neural_rollout_action(game, model, random.Random(1)).kind, 'throw')
            infer.assert_not_called()

    def test_search_uses_neural_followups_not_heuristic(self):
        game = Game.new(random.Random(7))
        def uniform(model, state, actions):
            return [1/len(actions)]*len(actions), 0.0
        with patch('yut_ai.player.evaluate', side_effect=uniform) as infer, patch('yut_ai.player.rollout_action', side_effect=AssertionError('heuristic used')):
            actions, policy, _ = search(game, object(), random.Random(1), simulations=4, depth_turns=None, rollout_steps=12)
            self.assertAlmostEqual(sum(policy), 1)
            self.assertGreater(infer.call_count, 1 + max(4, 4*len(actions)))

    def test_legacy_replay_cannot_resume(self):
        from yut_ai.selfplay import validate_resume
        from yut_ai.player import SEARCH_VERSION
        for saved in ({}, {'run': {}}, {'run': {}, 'search_version': 'old'}):
            with self.assertRaises(ValueError):
                validate_resume(saved)
        validate_resume({'run': {}, 'search_version': SEARCH_VERSION})

    def test_hidden_opponent_hand_does_not_change_search(self):
        import torch
        from yut_ai.model import ModelConfig, PolicyValueNet
        from yut_ai.player import STATE_DIM, ACTION_DIM
        torch.manual_seed(9)
        model = PolicyValueNet(ModelConfig(STATE_DIM, ACTION_DIM))
        game = Game.new(random.Random(5))
        other = game.clone()
        other.hands[1] = [(game.hands[0][0]+1)%8, (game.hands[0][1]+1)%8]
        args = dict(simulations=4, depth_turns=None, rollout_steps=8)
        self.assertEqual(search(game, model, random.Random(2), **args),
                         search(other, model, random.Random(2), **args))
