# Fixed assumptions for the synthetic four-region experiment
class EnvironmentConfig:
    def __init__(self):
        self.neighbors = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}
        self.base_emissions = (2, 5, 8)
        self.mitigation_fractions = (0.1, 0.3, 0.5)
        self.action_costs = (1, 4, 9)
        self.persistence = 0.8
        self.spillover = 0.2
        self.episode_steps = 50


class TrainingConfig:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_minimum = 0.05
        self.training_episodes = 2000
        self.training_seeds = (11, 23, 37, 41, 53)
        self.evaluation_seeds = tuple(range(100_000, 100_020))
