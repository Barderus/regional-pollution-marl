"""Four-region environment with separate air, water, soil, and noise stocks."""

import random

from study_2_economic_sustainability.config import EnvironmentConfig


ENVIRONMENT_CHOICES = (
    "none",
    "prevent_low",
    "prevent_medium",
    "prevent_high",
    "clean_low",
    "clean_medium",
    "clean_high",
)
ACTION_SPECS = [(0, 0), (1, 0)]
ACTION_NAMES = []
ACTION_NAMES.append("maintain_none")
ACTION_NAMES.append("develop_none")

for investment_choice in range(1, len(ENVIRONMENT_CHOICES)):
    ACTION_SPECS.append((0, investment_choice))
    ACTION_NAMES.append(f"maintain_{ENVIRONMENT_CHOICES[investment_choice]}")

for investment_choice in range(1, len(ENVIRONMENT_CHOICES)):
    ACTION_SPECS.append((1, investment_choice))
    ACTION_NAMES.append(f"develop_{ENVIRONMENT_CHOICES[investment_choice]}")


def action_parts(action):
    return ACTION_SPECS[action]


def value_bin(value, boundaries):
    if value < boundaries[0]:
        return 0
    if value < boundaries[1]:
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
        self.pollutant_loads = {}
        self.pollution_loads = ()
        self.global_pollution_index = 0.0
        self.step_count = 0
        self._started = False
        self._done = False

    def _regional_index(self, region):
        total = 0.0

        for pollutant in self.config.pollutants:
            normalized_load = self.pollutant_loads[pollutant][region]
            normalized_load /= self.config.pollution_limits[pollutant]
            weight = self.config.pollutant_weights[pollutant]
            total += weight * min(normalized_load, 1.0)

        return self.config.gpi_scale * total

    def _update_indices(self):
        regional_indices = []

        for region in self.config.regions:
            regional_indices.append(self._regional_index(region))

        self.pollution_loads = tuple(regional_indices)
        self.global_pollution_index = sum(regional_indices) / len(regional_indices)

    def _observations(self):
        observations = []

        for region in self.config.regions:
            observation = [
                value_bin(self.development_indices[region], (34, 67)),
                value_bin(self.currency_balances[region], (200, 500)),
            ]

            for pollutant in self.config.pollutants:
                load = self.pollutant_loads[pollutant][region]
                limit = self.config.pollution_limits[pollutant]
                observation.append(value_bin(load / limit, (0.33, 0.67)))

            observation.append(value_bin(self.global_pollution_index, (34, 67)))
            observations.append(tuple(observation))

        return tuple(observations)

    def reset(self, seed):
        self.rng.seed(seed)
        balances = []

        for _ in self.config.regions:
            balance = self.rng.randint(
                self.config.initial_currency_minimum,
                self.config.initial_currency_maximum,
            )
            balances.append(balance)

        self.currency_balances = tuple(balances)
        self.development_indices = (0, 0, 0, 0)
        self.pollutant_loads = {}

        for pollutant in self.config.pollutants:
            initial_load = self.config.initial_pollution[pollutant]
            self.pollutant_loads[pollutant] = (initial_load,) * len(self.config.regions)

        self.step_count = 0
        self._started = True
        self._done = False
        self._update_indices()

        return self._observations()

    def _investment(self, investment_choice):
        prevention_level = 0
        cleanup_level = 0
        cost = 0.0

        if investment_choice in (1, 2, 3):
            prevention_level = investment_choice
            cost = self.config.prevention_costs[prevention_level]
        elif investment_choice in (4, 5, 6):
            cleanup_level = investment_choice - 3
            cost = self.config.cleanup_costs[cleanup_level]

        return prevention_level, cleanup_level, cost

    def step(self, actions):
        if not self._started or self._done:
            raise RuntimeError("Reset the environment before stepping")

        if not isinstance(actions, (list, tuple)) or len(actions) != len(self.config.regions):
            raise ValueError("Actions must contain one valid action for each region")

        for action in actions:
            if type(action) is not int or action < 0 or action >= len(ACTION_NAMES):
                raise ValueError(f"Actions must be integers from 0 to {len(ACTION_NAMES) - 1}")

        config = self.config
        income = []
        costs_paid = []
        development_costs_paid = []
        prevention_costs_paid = []
        cleanup_costs_paid = []
        valid_actions = []
        development_gains = []
        prevention_levels = []
        cleanup_levels = []
        new_balances = []
        new_development = []

        # Resolve all regions' actions against the same start-of-step state.
        for region in config.regions:
            development_choice, investment_choice = action_parts(actions[region])
            prevention_level, cleanup_level, investment_cost = self._investment(investment_choice)
            current_edi = self.development_indices[region]
            current_balance = self.currency_balances[region]
            region_income = config.income_base + config.income_per_edi * current_edi

            development_cost = config.development_cost if development_choice else 0.0
            total_cost = development_cost + investment_cost
            can_develop = not development_choice or current_edi < 100
            is_valid = current_balance >= total_cost and can_develop
            paid_cost = total_cost if is_valid else 0.0
            paid_development_cost = development_cost if is_valid else 0.0
            paid_investment_cost = investment_cost if is_valid else 0.0
            gain = 0

            if is_valid and development_choice:
                gain = min(100 - current_edi, development_gain(current_edi))

            income.append(region_income)
            costs_paid.append(paid_cost)
            development_costs_paid.append(paid_development_cost)
            if investment_choice in (1, 2, 3) and is_valid:
                prevention_costs_paid.append(paid_investment_cost)
                cleanup_costs_paid.append(0.0)
            elif investment_choice in (4, 5, 6) and is_valid:
                prevention_costs_paid.append(0.0)
                cleanup_costs_paid.append(paid_investment_cost)
            else:
                prevention_costs_paid.append(0.0)
                cleanup_costs_paid.append(0.0)
            valid_actions.append(is_valid)
            development_gains.append(gain)
            prevention_levels.append(prevention_level if is_valid else 0)
            cleanup_levels.append(cleanup_level if is_valid else 0)
            new_balances.append(current_balance - paid_cost + region_income)
            new_development.append(current_edi + gain)

        gross_emissions = {}
        remaining_emissions = {}

        for pollutant in config.pollutants:
            gross_emissions[pollutant] = []
            remaining_emissions[pollutant] = []

            for region in config.regions:
                edi = self.development_indices[region]
                emissions = config.baseline_emissions[pollutant]
                emissions += config.edi_emissions_per_point[pollutant] * edi
                prevention_level = prevention_levels[region]
                prevention_fraction = config.prevention_fractions[prevention_level]
                prevented = emissions * prevention_fraction

                gross_emissions[pollutant].append(emissions)
                remaining_emissions[pollutant].append(emissions - prevented)

        # Transport is a redistribution: outgoing emissions are deducted from
        # the source and divided evenly among its neighbors.
        incoming_emissions = {}
        local_emissions = {}

        for pollutant in config.pollutants:
            incoming_emissions[pollutant] = []
            local_emissions[pollutant] = []
            transport_fraction = config.transport_fractions[pollutant]

            for region in config.regions:
                neighbors = config.neighbors[region]
                incoming = 0.0

                for neighbor in neighbors:
                    outgoing = remaining_emissions[pollutant][neighbor]
                    outgoing *= transport_fraction
                    outgoing /= len(config.neighbors[neighbor])
                    incoming += outgoing

                retained_fraction = 1.0 - transport_fraction
                local = remaining_emissions[pollutant][region] * retained_fraction
                incoming_emissions[pollutant].append(incoming)
                local_emissions[pollutant].append(local + incoming)

        new_pollutant_loads = {}
        natural_removal = {}
        cleanup_removed = {}
        incoming_pollution = {}

        for pollutant in config.pollutants:
            new_pollutant_loads[pollutant] = []
            natural_removal[pollutant] = []
            cleanup_removed[pollutant] = []
            recovered_stocks = []
            pollutant_natural_removal = []
            transport_fraction = config.transport_fractions[pollutant]

            for region in config.regions:
                old_load = self.pollutant_loads[pollutant][region]
                limit = config.pollution_limits[pollutant]
                severity = min(old_load / limit, 1.0)
                recovery_factor = 1.0 - config.recovery_degradation[pollutant] * severity
                recovery_factor = max(config.minimum_recovery_fraction, recovery_factor)
                removal_fraction = config.natural_recovery_rates[pollutant] * recovery_factor
                removed_naturally = min(old_load, old_load * removal_fraction)
                load_after_recovery = old_load - removed_naturally
                recovered_stocks.append(load_after_recovery)
                pollutant_natural_removal.append(removed_naturally)

            incoming_stocks_by_region = []
            pollution_after_transport = []

            for region in config.regions:
                incoming_stock = 0.0

                for neighbor in config.neighbors[region]:
                    outgoing_stock = recovered_stocks[neighbor] * transport_fraction
                    outgoing_stock /= len(config.neighbors[neighbor])
                    incoming_stock += outgoing_stock

                retained_stock = recovered_stocks[region] * (1.0 - transport_fraction)
                incoming_stocks_by_region.append(incoming_stock)
                pollution_after_transport.append(retained_stock + incoming_stock)

            for region in config.regions:
                load_before_cleanup = (
                    pollution_after_transport[region]
                    + local_emissions[pollutant][region]
                )

                cleanup_level = cleanup_levels[region]
                cleanup_capacity = config.cleanup_capacities[cleanup_level]
                cleanup_difficulty = config.cleanup_difficulty[pollutant]
                current_severity = min(
                    load_before_cleanup / config.pollution_limits[pollutant],
                    1.0,
                )
                cleanup_efficiency = 1.0 - cleanup_difficulty * current_severity
                cleanup_efficiency = max(config.minimum_cleanup_efficiency, cleanup_efficiency)
                cleanup_amount = min(load_before_cleanup, cleanup_capacity * cleanup_efficiency)

                natural_removal[pollutant].append(pollutant_natural_removal[region])
                cleanup_removed[pollutant].append(cleanup_amount)
                new_pollutant_loads[pollutant].append(load_before_cleanup - cleanup_amount)

            incoming_pollution[pollutant] = tuple(incoming_stocks_by_region)

        self.currency_balances = tuple(new_balances)
        self.development_indices = tuple(new_development)
        self.pollutant_loads = {}

        for pollutant in config.pollutants:
            self.pollutant_loads[pollutant] = tuple(new_pollutant_loads[pollutant])

        self._update_indices()

        local_rewards = []
        pollution_damage = []

        for region in config.regions:
            damage = 0.0

            for pollutant in config.pollutants:
                load = self.pollutant_loads[pollutant][region]
                limit = config.pollution_limits[pollutant]
                normalized_load = load / limit
                excess = max(0.0, normalized_load - config.damage_threshold_fraction)
                pollutant_damage = normalized_load + 4.0 * excess ** 2
                damage += config.pollutant_weights[pollutant] * pollutant_damage

            income_reward = (income[region] - config.income_base)
            income_reward /= config.ongoing_income_reward_scale
            economic_reward = development_gains[region] + income_reward
            local_reward = config.economy_weight * economic_reward
            local_reward -= config.pollution_weight * damage

            if not valid_actions[region]:
                local_reward -= config.invalid_action_penalty

            local_rewards.append(local_reward)
            pollution_damage.append(damage)

        global_reward = sum(local_rewards) / len(config.regions)
        mixed_rewards = []

        for local_reward in local_rewards:
            mixed_reward = 0.7 * local_reward + 0.3 * global_reward
            mixed_rewards.append(mixed_reward)

        if self.reward_mode == "local":
            rewards = list(local_rewards)
        elif self.reward_mode == "global":
            rewards = [global_reward] * len(config.regions)
        else:
            rewards = list(mixed_rewards)

        failed_regions = []

        for region in config.regions:
            for pollutant in config.pollutants:
                load = self.pollutant_loads[pollutant][region]
                if load >= config.pollution_limits[pollutant]:
                    failed_regions.append((region, pollutant))

        terminated = len(failed_regions) > 0
        self.step_count += 1
        truncated = self.step_count >= config.episode_steps and not terminated
        self._done = terminated or truncated

        if terminated:
            adjusted_rewards = []
            for reward in rewards:
                adjusted_rewards.append(reward - config.failure_penalty)
            rewards = adjusted_rewards

        info = {
            "attempted_actions": tuple(actions),
            "action_names": tuple(ACTION_NAMES[action] for action in actions),
            "valid_actions": tuple(valid_actions),
            "invalid_attempts": tuple(not valid for valid in valid_actions),
            "costs_paid": tuple(costs_paid),
            "development_costs_paid": tuple(development_costs_paid),
            "prevention_costs_paid": tuple(prevention_costs_paid),
            "cleanup_costs_paid": tuple(cleanup_costs_paid),
            "income": tuple(income),
            "development_gains": tuple(development_gains),
            "development_indices": self.development_indices,
            "currency_balances": self.currency_balances,
            "gross_emissions": gross_emissions,
            "remaining_emissions": remaining_emissions,
            "incoming_emissions": incoming_emissions,
            "local_emissions": local_emissions,
            "natural_removal": natural_removal,
            "incoming_pollution": incoming_pollution,
            "cleanup_removed": cleanup_removed,
            "pollutant_loads": self.pollutant_loads,
            "regional_pollution_indices": self.pollution_loads,
            "pollution_loads": self.pollution_loads,
            "global_pollution_index": self.global_pollution_index,
            "pollution_damage": tuple(pollution_damage),
            "failed_regions": tuple(failed_regions),
            "local_rewards": tuple(local_rewards),
            "global_reward": global_reward,
            "mixed_rewards": tuple(mixed_rewards),
        }

        return self._observations(), tuple(rewards), terminated, truncated, info
