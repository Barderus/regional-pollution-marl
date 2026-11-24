"""Tabular Q-learning agent used after the environment is calibrated."""

import random


class TabularQLearner:
    def __init__(self, action_count, seed, learning_rate=0.1,
                 discount_factor=0.95, epsilon=1.0):
        self.action_count = action_count
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        self.q_values = {}

    def _state_values(self, state):
        if state not in self.q_values:
            self.q_values[state] = [0.0] * self.action_count

        return self.q_values[state]

    def select_action(self, state, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_count)

        values = self._state_values(state)
        best_action = 0

        for action in range(1, self.action_count):
            if values[action] > values[best_action]:
                best_action = action

        return best_action

    def update(self, state, action, reward, next_state, terminal):
        values = self._state_values(state)

        if terminal:
            target = reward
        else:
            target = reward + self.discount_factor * max(
                self._state_values(next_state)
            )

        values[action] += self.learning_rate * (target - values[action])

    def decay_epsilon(self, decay_rate, minimum):
        self.epsilon = max(minimum, self.epsilon * decay_rate)
