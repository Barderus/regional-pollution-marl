class EnvironmentConfig:
    def __init__(self):
        self.neighbors = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}

        # Each episode starts with no economic development and these resources.
        self.initial_pollution = 10.0
        self.initial_currency_minimum = 100
        self.initial_currency_maximum = 1000

        # EDI means Economic Development Index (0 to 100).
        # Income = 20 + 0.5 * EDI currency per step.
        # Base emissions = 8 * (1 + EDI / 100) arbitrary units per step.
        self.income_base = 20.0
        self.income_per_edi = 0.5
        self.emissions_base = 8.0

        # Action IDs: 0 save, 1 develop, 2 low mitigation, 3 medium mitigation, 4 high mitigation.
        self.action_costs = (0, 100, 10, 25, 50)
        self.mitigation_capacity = (0, 0, 5, 15, 30)

        # Eighty percent of old pollution remains each step. Each neighbor's remaining emissions add another 20 percent to a region's load.
        self.persistence = 0.8
        self.spillover = 0.2

        # Any region reaching 100 ends the episode. All agents then receive the failure penalty. An invalid attempt penalizes its own region.
        self.failure_threshold = 100.0
        self.failure_penalty = 100.0
        self.invalid_action_penalty = 10.0

        # Local reward combines the EDI gain with -(new pollution / 20)^2.
        self.economy_weight = 1.0
        self.pollution_weight = 1.0

        # Truncation
        self.episode_steps = 200

class TrainingConfig:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_minimum = 0.05
