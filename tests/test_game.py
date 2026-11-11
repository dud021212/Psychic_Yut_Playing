import random
import unittest
from yut_ai.game import Game, Action, HOME, DONE, paths
from yut_ai.rules import MoveResult, Power
from yut_ai.player import features, STATE_DIM, action_features, ACTION_DIM

class GameTests(unittest.TestCase):
    def game(self):
        return Game.new(random.Random(2))

    def test_routes(self):
        self.assertEqual(paths(5, MoveResult('geol', 3)), [(20, 21, 22)])
        self.assertEqual(paths(4, MoveResult('gae', 2)), [(5, 6)])
        self.assertEqual(paths(22, MoveResult('yut', 4)), [(27, 28, 0, DONE)])
        self.assertEqual(paths(10, MoveResult('yut', 4)), [(25, 26, 22, 27)])
        self.assertEqual(paths(10, MoveResult('mo', 5)), [(25, 26, 22, 27, 28)])
        self.assertEqual(paths(25, MoveResult('yut', 4)), [(26, 22, 27, 28)])
        self.assertEqual(paths(26, MoveResult('geol', 3)), [(22, 27, 28)])
        self.assertEqual(paths(5, MoveResult('mo', 5)), [(20, 21, 22, 23, 24)])
        self.assertEqual(set(paths(22, MoveResult('backdo', -1))), {(21,), (26,)})
        self.assertEqual(paths(2, MoveResult('mo', -5, True)), [(1, 0)])
        self.assertEqual(len(paths(0, MoveResult('mo', -5, True))), 3)

    def test_gate_capture_and_discard(self):
        g = self.game()
        g.phase = 'move'
        g.positions = [[1, HOME, HOME, HOME], [0, 0, HOME, HOME]]
        g.moves = [MoveResult('geol', 3)]
        g.installations = {2: 'gate'}
        g.apply(g.legal_actions()[0], random.Random(0))
        self.assertEqual(g.positions[0][0], 0)
        self.assertEqual(g.positions[1][:2], [HOME, HOME])
        self.assertTrue(g.extra)
        self.assertFalse(g.installations)
        g = self.game(); g.phase = 'move'
        g.moves = [MoveResult('backdo', -1), MoveResult('yut', 4)]
        self.assertTrue(all(a.args[0] == 1 for a in g.legal_actions()))
        g.apply(g.legal_actions()[0], random.Random(0))
        self.assertTrue(any(a.kind == 'move' and a.args[0] == 0 for a in g.legal_actions()))

    def test_takeout_and_stacked_finish(self):
        g = self.game()
        g.phase = 'move'; g.takeout = True
        g.moves = [MoveResult('mo', 5)]
        g.apply(Action('end'), random.Random(0))
        self.assertEqual(len(g.stored[0]), 1)
        self.assertFalse(g.extra)
        g.player = 0; g.phase = 'throw'
        g.apply(Action('throw'), random.Random(0))
        self.assertFalse(g.stored[0])
        self.assertIn(MoveResult('mo', 5), g.moves)
        g = self.game(); g.phase = 'move'
        g.positions[0] = [0,0,0,0]; g.moves = [MoveResult('do', 1)]
        g.apply(g.legal_actions()[0], random.Random(0))
        self.assertEqual(g.positions[0], [DONE]*4)
        self.assertEqual(g.winner, 0)

    def test_hidden_input(self):
        g = self.game()
        before = features(g)
        g.hands[1] = [(g.hands[0][0]+1)%8, (g.hands[0][1]+1)%8]
        self.assertEqual(before, features(g))
        self.assertEqual(len(before), STATE_DIM)
        self.assertTrue(all(len(action_features(g,a)) == ACTION_DIM for a in g.legal_actions()))

    def test_all_powers_and_complete_games(self):
        for power in range(8):
            for category in (0, 1):
                g = self.game()
                g.hands[0][category] = power
                g.hands[1][category] = (power+1)%8
                from yut_ai.belief import PowerBelief
                g.beliefs = [PowerBelief(*h) for h in g.hands]
                g.positions = [[1,1,HOME,HOME], [3,3,10,HOME]]
                actions = [a for a in g.legal_actions() if a.kind == 'power' and a.args[0] == category]
                self.assertTrue(actions, (category,power))
                for a in actions:
                    trial = g.clone(); trial.apply(a, random.Random(0)); trial.check()
        from yut_ai.player import rollout_action
        for seed in range(10):
            rng = random.Random(seed); g = Game.new(rng)
            for _ in range(2000):
                g.apply(rollout_action(g,rng), rng)
                if g.winner is not None: break
            self.assertIsNotNone(g.winner, seed)

if __name__ == '__main__': unittest.main()
