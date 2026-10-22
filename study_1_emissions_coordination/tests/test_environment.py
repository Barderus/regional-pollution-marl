"""Focused checks of the documented environment equations and step order."""

import unittest

from baselines import fixed_medium_action, threshold_action
from environment import PollutionEnvironment, pollution_bin


class EnvironmentTests(unittest.TestCase):
    def test_simultaneous_emissions_and_spillover(self):
        env = PollutionEnvironment()
        env.reset(7)
        env.demand = (0, 1, 2, 1)
        _, _, _, info = env.step((0, 1, 2, 0))
        emissions = (1.8, 3.5, 4.0, 4.5)
        for actual, expected in zip(info["emissions"], emissions):
            self.assertAlmostEqual(actual, expected)
        expected_loads = (
            16 + 1.8 + 0.2 * (3.5 + 4),
            16 + 3.5 + 0.2 * (1.8 + 4.5),
            16 + 4 + 0.2 * (1.8 + 4.5),
            16 + 4.5 + 0.2 * (3.5 + 4),
        )
        for actual, expected in zip(info["pollution_loads"], expected_loads):
            self.assertAlmostEqual(actual, expected)

    def test_reward_and_demand_update_order(self):
        env = PollutionEnvironment("mixed")
        env.reset(13)
        env.demand = (2, 2, 2, 2)
        next_obs, rewards, _, info = env.step((0, 1, 2, 0))
        self.assertEqual(info["mitigation_costs"], (1, 4, 9, 1))
        local = tuple(-x - 0.2 * max(0, x - 30) ** 2 - 2 * cost
                      for x, cost in zip(info["pollution_loads"], info["mitigation_costs"]))
        for actual, expected in zip(info["local_rewards"], local):
            self.assertAlmostEqual(actual, expected)
        global_reward = sum(local) / 4
        self.assertAlmostEqual(info["global_reward"], global_reward)
        for i in range(4):
            self.assertAlmostEqual(rewards[i], 0.7 * local[i] + 0.3 * global_reward)
            self.assertEqual(next_obs[i], (env.demand[i], pollution_bin(env.loads[i])))

    def test_action_validation_and_demand_bounds(self):
        env = PollutionEnvironment()
        with self.assertRaises(RuntimeError):
            env.step((0, 0, 0, 0))
        env.reset(3)
        for actions in ((0, 1), (0, 1, 2, 3), (0, 1, True, 2)):
            with self.assertRaises(ValueError):
                env.step(actions)
        self.assertEqual(env.time, 0)
        for _ in range(50):
            env.step((0, 1, 2, 0))
            self.assertTrue(all(0 <= d <= 2 for d in env.demand))

    def test_seeded_replay_and_terminal_step(self):
        first = PollutionEnvironment()
        second = PollutionEnvironment()
        self.assertEqual(first.reset(89), second.reset(89))
        for step in range(50):
            a = first.step((1, 1, 1, 1))
            b = second.step((1, 1, 1, 1))
            self.assertEqual(a, b)
            self.assertEqual(a[2], step == 49)
        with self.assertRaises(RuntimeError):
            first.step((1, 1, 1, 1))

    def test_bins_and_baselines(self):
        self.assertEqual([pollution_bin(x) for x in (14.99, 15, 24.99, 25, 34.99, 35)],
                         [0, 1, 1, 2, 2, 3])
        self.assertEqual([threshold_action((0, b)) for b in range(4)], [0, 0, 1, 2])
        self.assertEqual(fixed_medium_action((2, 3)), 1)


if __name__ == "__main__":
    unittest.main()
