import random
from study_2_economic_sustainability.config import TrainingConfig

class TabularQLearner:
    def __init__(self, seed, config=None):
        self.config = config or TrainingConfig()
        self.rng = random.Random(seed)
        self.epsilon = self.config.epsilon_start

        # Create four learners to give the regions separate Q-tables.
        self.q = {}

    def _values(self, observation):
        if observation not in self.q:
            self.q[observation] = [0.0, 0.0, 0.0, 0.0, 0.0]

        return self.q[observation]

    def select_action(self, observation, greedy=False):
        # Choose among all five actions, even those the region cannot afford
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(5)

        values = self._values(observation)
        best_action = 0

        # Strict comparison keeps the lowest action ID when values are tied
        for action in range(1, 5):
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
