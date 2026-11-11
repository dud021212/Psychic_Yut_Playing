import random
import unittest
from yut_ai.game import Game
from yut_ai.evaluate import evaluation_jobs, evaluate_game

class EvaluationGamesTests(unittest.TestCase):
    def test_independent_jobs_and_reproducibility(self):
        jobs = evaluation_jobs(100000, 101, 8, 16)
        self.assertEqual(len(jobs), 101)
        self.assertEqual(jobs, evaluation_jobs(100000, 101, 8, 16))
        self.assertNotEqual(jobs, evaluation_jobs(200000, 101, 8, 16))
        self.assertEqual(len({job[0] for job in jobs}), 101)
        self.assertEqual({job[1] for job in jobs}, {0, 1})
        hands = set()
        for seed, seat, simulations, steps in jobs:
            game = Game.new(random.Random(seed))
            hands.add(tuple(tuple(h) for h in game.hands))
            for owner in (0, 1):
                self.assertEqual(tuple(game.hands[owner]), game.beliefs[owner].own)
            self.assertEqual((simulations, steps), (8, 16))
        self.assertGreater(len(hands), 50)

    def test_game_job_replays_exactly(self):
        for job in evaluation_jobs(7, 3, 8, 16):
            result = evaluate_game(None, None, job)
            self.assertIsInstance(result, bool)
            self.assertEqual(result, evaluate_game(None, None, job))

if __name__ == '__main__': unittest.main()
