"""
    Four neighboring regions choose how to develop and reduce pollution.
    0 -- 1
    |    |
    2 -- 3
"""


import random
from study_2_economic_sustainability.config import EnvironmentConfig

DEVELOPMENT_ACTION = 1


def observation_bin(value):
    if value < 34:
        return 0
    if value < 67:
        return 1
    return 2


def currency_bin(balance):
    if balance < 200:
        return 0
    if balance < 500:
        return 1
    return 2


def development_gain(current_edi):
    if current_edi < 40:
        return 10
    if current_edi < 70:
        return 5
    if current_edi < 100:
        return 2
    return 0


class EconomicSustainabilityEnvironment:
    def __init__(self, reward_mode="local", config=None):
        if reward_mode not in ("local", "global", "mixed"):
            raise ValueError(f"Unknown reward mode: {reward_mode}")

        self.reward_mode = reward_mode
        self.config = config or EnvironmentConfig()
        self.rng = random.Random()

        self.currency_balances = ()
        self.development_indices = ()
        self.pollution_loads = ()
        self.global_pollution_index = 0.0
        self.step_count = 0
        self._started = False
        self._done = False

    def _observations(self):
        # Each region sees its own bins and the shared pollution index bin
        observations = []

        for region in range(4):
            observation = (
                observation_bin(self.development_indices[region]),
                currency_bin(self.currency_balances[region]),
                observation_bin(self.pollution_loads[region]),
                observation_bin(self.global_pollution_index))
            observations.append(observation)

        return tuple(observations)

    def reset(self, seed):
        self.rng.seed(seed)

        balances = []
        for _ in range(4):
            balance = self.rng.randint(
                self.config.initial_currency_minimum,
                self.config.initial_currency_maximum,
            )
            balances.append(balance)

        self.currency_balances = tuple(balances)
        self.development_indices = (0, 0, 0, 0)
        self.pollution_loads = (self.config.initial_pollution,) * 4
        self.global_pollution_index = self.config.initial_pollution
        self.step_count = 0
        self._started = True
        self._done = False

        return self._observations()

    def step(self, actions):
        if not self._started or self._done:
            raise RuntimeError("Reset the environment before stepping")

        if not isinstance(actions, (list, tuple)) or len(actions) != 4:
            raise ValueError("Actions must be four integers in {0, 1, 2, 3, 4}")
        for action in actions:
            if type(action) is not int or action not in (0, 1, 2, 3, 4):
                raise ValueError("Actions must be four integers in {0, 1, 2, 3, 4}")

        config = self.config

        # 1. Use the state at the start of the step for income, emissions, affordability, and development gains.
        income = []
        base_emissions = []
        valid_actions = []
        costs_paid = []
        mitigation = []
        development_gains = []
        new_balances = []
        new_development = []

        for region in range(4):
            current_edi = self.development_indices[region]
            current_balance = self.currency_balances[region]
            action = actions[region]

            region_income = config.income_base + config.income_per_edi * current_edi
            region_emissions = config.emissions_base * (1 + current_edi / 100)
            can_pay = current_balance >= config.action_costs[action]
            can_develop = action != DEVELOPMENT_ACTION or current_edi < 100
            is_valid = can_pay and can_develop

            paid = config.action_costs[action] if is_valid else 0
            capacity = config.mitigation_capacity[action] if is_valid else 0
            gain = 0

            if action == DEVELOPMENT_ACTION and is_valid:
                gain = min(100 - current_edi, development_gain(current_edi))

            income.append(region_income)
            base_emissions.append(region_emissions)
            valid_actions.append(is_valid)
            costs_paid.append(paid)
            mitigation.append(capacity)
            development_gains.append(gain)
            new_balances.append(current_balance - paid + region_income)
            new_development.append(current_edi + gain)

        # 2. Purchased mitigation offsets new emissions first. Any unused capacity can remove old pollution after natural absorption.
        remaining_emissions = []
        unused_mitigation = []

        for region in range(4):
            remaining_emissions.append(max(0.0, base_emissions[region] - mitigation[region]))
            unused_mitigation.append(max(0.0, mitigation[region] - base_emissions[region]))

        # 3. Calculate every new load from the old loads and the same set of post-mitigation emissions.
        incoming_spillover = []
        new_loads = []

        for region in range(4):
            neighbor_emissions = sum(remaining_emissions[neighbor] for neighbor in config.neighbors[region])
            spillover = config.spillover * neighbor_emissions
            old_pollution_left = max(0.0, config.persistence * self.pollution_loads[region] - unused_mitigation[region])
            new_load = old_pollution_left + remaining_emissions[region] + spillover

            incoming_spillover.append(spillover)
            new_loads.append(new_load)

        new_global_index = sum(new_loads) / 4

        # 4. An invalid attempt costs no currency but lowers its own local reward
        pollution_rewards = []
        local_rewards = []

        for region in range(4):
            pollution_reward = -(new_loads[region] / 20) ** 2
            local_reward = (config.economy_weight * development_gains[region] + config.pollution_weight * pollution_reward)
            if not valid_actions[region]:
                local_reward -= config.invalid_action_penalty

            pollution_rewards.append(pollution_reward)
            local_rewards.append(local_reward)

        global_reward = sum(local_rewards) / 4
        mixed_rewards = [0.7 * local_reward + 0.3 * global_reward for local_reward in local_rewards]

        if self.reward_mode == "local":
            rewards = list(local_rewards)
        elif self.reward_mode == "global":
            rewards = [global_reward] * 4
        else:
            rewards = list(mixed_rewards)

        # 5. One region reaching 100 ends the episode for everyone. Apply the harsh penalty. The episode truncates at 200 steps with no penalty.
        terminated = any(load >= config.failure_threshold for load in new_loads)
        self.step_count += 1
        truncated = self.step_count >= config.episode_steps and not terminated
        self._done = terminated or truncated

        if terminated:
            rewards = [reward - config.failure_penalty for reward in rewards]

        self.currency_balances = tuple(new_balances)
        self.development_indices = tuple(new_development)
        self.pollution_loads = tuple(new_loads)
        self.global_pollution_index = new_global_index

        info = {
            "attempted_actions": tuple(actions),
            "valid_actions": tuple(valid_actions),
            "invalid_attempts": tuple(not valid for valid in valid_actions),
            "costs_paid": tuple(costs_paid),
            "income": tuple(income),
            "development_gains": tuple(development_gains),
            "currency_balances": self.currency_balances,
            "development_indices": self.development_indices,
            "base_emissions": tuple(base_emissions),
            "mitigation_capacity": tuple(mitigation),
            "remaining_emissions": tuple(remaining_emissions),
            "unused_mitigation": tuple(unused_mitigation),
            "incoming_spillover": tuple(incoming_spillover),
            "pollution_loads": self.pollution_loads,
            "global_pollution_index": self.global_pollution_index,
            "economic_rewards": tuple(development_gains),
            "pollution_rewards": tuple(pollution_rewards),
            "local_rewards": tuple(local_rewards),
            "global_reward": global_reward,
            "mixed_rewards": tuple(mixed_rewards),
        }

        return self._observations(), tuple(rewards), terminated, truncated, info
