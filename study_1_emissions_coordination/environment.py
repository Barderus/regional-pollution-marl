# Simultaneous mitigation and persistent pollution in arbitrary units

import random
from study_1_emissions_coordination.config import EnvironmentConfig

def pollution_bin(load):
    if load < 15:
        return 0
    if load < 25:
        return 1
    if load < 35:
        return 2
    return 3


def exposure_penalty(load):
    return load + 0.2 * max(0.0, load - 30) ** 2

# Four regions with independent, seeded demand changes
class PollutionEnvironment:

    def __init__(self, reward_mode="local", config=None, externality_weight=0.0):
        if reward_mode not in ("local", "global", "mixed", "externality"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")
        if externality_weight < 0 or (reward_mode != "externality" and externality_weight != 0):
            raise ValueError("Externality weight requires externality mode and cannot be negative")
        self.reward_mode = reward_mode
        self.externality_weight = externality_weight
        self.config = config or EnvironmentConfig()
        self.rng = random.Random()
        self.demand = ()
        self.loads = ()
        self.time = 0
        self._started = False

    def _observations(self):
        return tuple((d, pollution_bin(x)) for d, x in zip(self.demand, self.loads))

    def reset(self, seed):
        self.rng.seed(seed)
        self.time = 0
        self.loads = (20.0,) * 4
        self.demand = tuple(self.rng.randrange(3) for _ in range(4))
        self._started = True
        return self._observations()

    def step(self, actions):
        if not self._started or self.time >= self.config.episode_steps:
            raise RuntimeError("Reset the environment before stepping")
        if len(actions) != 4 or any(type(a) is not int or a not in (0, 1, 2) for a in actions):
            raise ValueError("Actions must be four integers in {0, 1, 2}")
        config = self.config

        # All regions act on the same current demand and pollution loads.
        emissions = tuple(
            config.base_emissions[demand] * (1 - config.mitigation_fractions[action])
            for demand, action in zip(self.demand, actions)
        )

        new_loads = tuple(
            config.persistence * self.loads[region] + emissions[region]
            + config.spillover * sum(emissions[neighbor] for neighbor in config.neighbors[region])
            for region in range(4)
        )

        costs = tuple(config.action_costs[action] for action in actions)
        exposure = tuple(exposure_penalty(load) for load in new_loads)
        local_rewards = tuple(-penalty - 2 * cost for penalty, cost in zip(exposure, costs))
        global_reward = sum(local_rewards) / 4
        mixed_rewards = tuple(0.7 * reward + 0.3 * global_reward for reward in local_rewards)
        rewards_by_mode = {
            "local": local_rewards,
            "global": (global_reward,) * 4,
            "mixed": mixed_rewards,
        }

        if self.reward_mode == "externality":
            # Remove this region's spillover from each neighbor's load to measure its harm.
            neighbor_harm = tuple(
                sum(exposure[neighbor] - exposure_penalty(
                    new_loads[neighbor] - config.spillover * emissions[region]
                ) for neighbor in config.neighbors[region])
                for region in range(4)
            )
            rewards = tuple(local_rewards[region] - self.externality_weight * neighbor_harm[region]
                            for region in range(4))
        else:
            rewards = rewards_by_mode[self.reward_mode]
        self.loads = new_loads

        # Demand changes only after rewards are calculated, for the next observation.
        self.demand = tuple(
            max(0, min(2, demand + self.rng.choice((-1, 0, 1))))
            for demand in self.demand
        )

        self.time += 1
        info = {
            "emissions": emissions,
            "pollution_loads": new_loads,
            "mitigation_costs": costs,
            "exposure_penalties": exposure,
            "local_rewards": local_rewards,
            "global_reward": global_reward,
            "mixed_rewards": mixed_rewards,
        }
        if self.reward_mode == "externality":
            info["neighbor_harm"] = neighbor_harm
            info["externality_rewards"] = rewards
        return self._observations(), rewards, self.time == config.episode_steps, info
