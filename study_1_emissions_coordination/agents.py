# Independent tabular Q-learning with local observations

import random
from study_1_emissions_coordination.config import TrainingConfig


class TabularQLearner:
    """Greedy ties choose the lowest action index, including during evaluation."""

    def __init__(self, seed, config=None):
        self.rng = random.Random(seed)
        self.config = config or TrainingConfig()
        self.epsilon = self.config.epsilon_start
        self.q = {}

    def _values(self, observation):
        return self.q.setdefault(observation, [0.0, 0.0, 0.0])

    def select_action(self, observation, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(3)
        values = self._values(observation)
        return max(range(3), key=lambda action: values[action])

    def update(self, observation, action, reward, next_observation, done):
        values = self._values(observation)
        if done:
            target = reward
        else:
            target = reward + self.config.discount_factor * max(self._values(next_observation))
        values[action] += self.config.learning_rate * (target - values[action])

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_minimum, self.epsilon * self.config.epsilon_decay)
