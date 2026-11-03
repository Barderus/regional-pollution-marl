class EnvironmentConfig:
    def __init__(self):
        self.regions = (0, 1, 2, 3)
        self.neighbors = {
            0: (1, 2),
            1: (0, 3),
            2: (0, 3),
            3: (1, 2),
        }

        # One step is provisionally treated as one year. Values remain synthetic.
        self.time_step_name = "year"
        self.episode_steps = 200

        self.pollutants = ("air", "water", "soil", "noise")
        self.pollutant_weights = {
            "air": 0.35,
            "water": 0.30,
            "soil": 0.25,
            "noise": 0.10,
        }

        # Baseline emissions occur at every development level. EDI adds activity.
        self.baseline_emissions = {
            "air": 3.0,
            "water": 1.0,
            "soil": 0.5,
            "noise": 2.0,
        }
        self.edi_emissions_per_point = {
            "air": 0.08,
            "water": 0.04,
            "soil": 0.02,
            "noise": 0.03,
        }

        # Fractions of current emissions and recovered pollution stock
        # transferred to neighboring regions. Transferred amounts leave the
        # source, so transport does not duplicate pollution.
        self.transport_fractions = {
            "air": 0.30,
            "water": 0.40,
            "soil": 0.10,
            "noise": 0.20,
        }

        # Fraction of a pollutant stock naturally removed during one time step.
        self.natural_recovery_rates = {
            "air": 0.60,
            "water": 0.12,
            "soil": 0.02,
            "noise": 0.90,
        }
        self.recovery_degradation = {
            "air": 0.25,
            "water": 0.70,
            "soil": 0.80,
            "noise": 0.0,
        }
        self.minimum_recovery_fraction = 0.20

        # Synthetic reference limits define normalized pollution and failure.
        # They are not legal standards or calibrated environmental thresholds.
        self.pollution_limits = {
            "air": 100.0,
            "water": 100.0,
            "soil": 100.0,
            "noise": 100.0,
        }
        self.initial_pollution = {
            "air": 10.0,
            "water": 10.0,
            "soil": 10.0,
            "noise": 10.0,
        }

        # Development keeps its earlier cost and diminishing EDI returns.
        self.development_cost = 100.0
        self.initial_currency_minimum = 100
        self.initial_currency_maximum = 1000
        self.income_base = 20.0
        self.income_per_edi = 0.5

        # Each region selects one environmental investment per year.
        # Prevention reduces new emissions. Cleanup acts on existing stocks.
        self.prevention_fractions = (0.0, 0.25, 0.50, 0.75)
        self.cleanup_capacities = (0.0, 2.0, 5.0, 10.0)
        self.prevention_costs = (0.0, 10.0, 25.0, 50.0)
        self.cleanup_costs = (0.0, 15.0, 40.0, 80.0)
        self.cleanup_difficulty = {
            "air": 0.0,
            "water": 0.50,
            "soil": 0.65,
            "noise": 0.0,
        }
        self.minimum_cleanup_efficiency = 0.15

        # GPI is a weighted average of each pollutant's normalized load.
        self.gpi_scale = 100.0
        self.damage_threshold_fraction = 0.50
        self.failure_penalty = 100.0
        self.invalid_action_penalty = 10.0

        self.ongoing_income_reward_scale = 10.0
        self.economy_weight = 1.0
        self.pollution_weight = 1.0


class TrainingConfig:
    def __init__(self):
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon_start = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_minimum = 0.05
        self.training_episodes = 2000
        self.training_seeds = (11, 23, 37, 41, 53)
        self.evaluation_seeds = tuple(range(200000, 200020))
