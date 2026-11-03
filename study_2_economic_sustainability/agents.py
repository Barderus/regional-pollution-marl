import random
from study_2_economic_sustainability.config import TrainingConfig
from study_2_economic_sustainability.environment import ACTION_NAMES

class TabularQLearner:
    def __init__(self, seed, config=None):
        self.config = config or TrainingConfig()
        self.rng = random.Random(seed)
        self.epsilon = self.config.epsilon_start

        self.q = {}

    def _values(self, observation):
        if observation not in self.q:
            self.q[observation] = [0.0] * len(ACTION_NAMES)

        return self.q[observation]

    def select_action(self, observation, greedy=False):
        # Exploration can include unaffordable choices; the environment records
        # and penalizes those attempts consistently across reward modes.
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(len(ACTION_NAMES))

        values = self._values(observation)
        best_action = 0

        # Strict comparison keeps the lowest action ID when values are tied
        for action in range(1, len(ACTION_NAMES)):
            if values[action] > values[best_action]:
                best_action = action

        return best_action

    def update(self, observation, action, reward, next_observation, terminated):
        values = self._values(observation)

        if terminated:
            target = reward
        else:
            next_values = self._values(next_observation)
            target = reward + self.config.discount_factor * max(next_values)

        old_value = values[action]
        values[action] = old_value + self.config.learning_rate * (target - old_value)

    def decay_epsilon(self):
        # Reduce exploration once after an episode, down to the minimum
        self.epsilon = max(
            self.config.epsilon_minimum,
            self.epsilon * self.config.epsilon_decay,
        )
