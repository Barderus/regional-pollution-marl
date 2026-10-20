"""Hand-calculated Q-learning checks."""

import unittest

from agents import TabularQLearner


class AgentTests(unittest.TestCase):
    def test_hand_calculated_q_update(self):
        agent = TabularQLearner(1)
        agent.q[(1, 2)] = [0.0, 10.0, 2.0]
        agent.q[(2, 3)] = [3.0, 5.0, 4.0]
        agent.update((1, 2), 1, -4.0, (2, 3), False)
        self.assertAlmostEqual(agent.q[(1, 2)][1], 10 + 0.1 * (-4 + 0.95 * 5 - 10))
        self.assertEqual(agent.select_action((2, 3), greedy=True), 1)
        self.assertEqual(agent.select_action((0, 0), greedy=True), 0)

    def test_terminal_update_without_bootstrap(self):
        agent = TabularQLearner(2)
        agent.q[(2, 3)] = [100.0, 100.0, 100.0]
        agent.update((0, 1), 2, -8.0, (2, 3), True)
        self.assertAlmostEqual(agent.q[(0, 1)][2], -0.8)


if __name__ == "__main__":
    unittest.main()
