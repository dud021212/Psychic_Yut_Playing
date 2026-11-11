import random
import unittest
from yut_ai.rules import *
from yut_ai.belief import PowerBelief
from yut_ai.observation import PlayerObservation
from yut_ai.runtime import cutoff

class RulesTests(unittest.TestCase):
    def test_distributions(self):
        for power in [None, *Power]:
            for first in (False, True):
                distribution = throw_distribution(power, first_throw=first)
                self.assertAlmostEqual(sum(distribution.values()), 1)
                exempt = first and power in SPECIAL
                self.assertEqual('nak' not in distribution, exempt)
                if not first:
                    self.assertEqual(distribution, throw_distribution())

    def test_effect_lifetime(self):
        self.assertEqual(len(resolve_throw('mo', Power.DOUBLE).moves), 2)
        self.assertEqual(len(resolve_throw('mo', Power.DOUBLE, first_throw=False).moves), 1)
        self.assertEqual(resolve_throw('backdo', Power.BACKSTEP).moves[0].distance, 1)
        self.assertEqual(resolve_throw('backdo', Power.BACKSTEP, first_throw=False).moves[0].distance, -1)
        self.assertEqual(resolve_throw('nak', Power.DOUBLE).moves, ())
        with self.assertRaises(ValueError):
            resolve_throw('nak', Power.MIDAS)

    def test_path(self):
        move = resolve_throw('mo', Power.BACKSTEP).moves[0]
        self.assertEqual(resolve_path([1, 0, 9, 8, 7], finish=0, start=2, move=move), (0, 'backstep_stop'))
        self.assertEqual(resolve_path([9, 8, 7, 6, 5], finish=0, start=0, move=move), (5, 'normal'))
        forward = resolve_throw('mo').moves[0]
        self.assertEqual(resolve_path([1, 2, 3, 4, 5], finish=0, start=9, move=forward, gate=2), (0, 'gate'))
        self.assertEqual(resolve_path([1, 2, 3], finish=0, start=9, move=forward, bomb=1, gate=2), (None, 'bomb'))
        self.assertFalse(extra_after_capture(False, 'mo'))
        self.assertTrue(extra_after_capture(False, 'backdo'))
        self.assertTrue(extra_after_capture(True, 'yut'))

    def test_backdo_shuffle_depth(self):
        self.assertFalse(discard_unusable_backdo(on_board=False, has_forward=True, extra=False))
        self.assertFalse(discard_unusable_backdo(on_board=False, has_forward=False, extra=True))
        self.assertTrue(discard_unusable_backdo(on_board=False, has_forward=False, extra=False))
        places = shuffle_groups(['a', 'b', 'c'], range(10), random.Random(1))
        self.assertEqual(len(set(places.values())), 3)
        self.assertFalse(cutoff('easy', 3))
        self.assertTrue(cutoff('easy', 4))
        self.assertTrue(cutoff('hard', 16))
        self.assertFalse(cutoff('expert', 100000))

    def test_invalid_inputs_do_not_create_impossible_beliefs(self):
        for value in (1.5, True, -1, 8):
            with self.assertRaises(ValueError):
                PowerBelief(value, 2)
        belief = PowerBelief(1, 2)
        before = belief.features()
        with self.assertRaises(ValueError):
            belief.observe_used_power(0, 3.5)
        self.assertEqual(before, belief.features())
        with self.assertRaises(ValueError):
            resolve_path([None], finish=0, start=1, move=resolve_throw('do').moves[0])

    def test_belief_observation(self):
        belief = PowerBelief(1, 2)
        self.assertEqual(sum(p > 0 for p in belief.features()), 49)
        belief.observe_used_power(0, 4)
        self.assertEqual(sum(p > 0 for p in belief.features()), 7)
        before = belief.features()
        with self.assertRaises(ValueError):
            belief.observe_used_power(1, 2)
        self.assertEqual(before, belief.features())
        belief.update([1.0] * 64)
        self.assertAlmostEqual(sum(belief.features()), 1)
        observation = PlayerObservation.from_belief([0, 1], belief)
        self.assertEqual(len(observation.encode()), 82)
        self.assertEqual(belief.sample(random.Random(0))[0], 4)

if __name__ == '__main__':
    unittest.main()
