from copy import deepcopy
import numpy as np
import pandas as pd

from study_2_economic_sustainability.config import EnvironmentConfig
from study_2_economic_sustainability.environment import (ACTION_NAMES, EconomicSustainabilityEnvironment, action_parts)


COMPARISON_SEEDS = tuple(range(201, 206))
COMPARISON_CASH = (150, 500, 1000)
COMPARISON_POLLUTION = (10.0, 50.0)
COMPARISON_POLICIES = (
    "sequential_high_prevention",
    "develop_medium_then_high",
    "threshold_adaptive",
    "original_reward_greedy",
    "proposed_reward_greedy",
)
CASH_OFFSETS = (-30, -10, 10, 30)


def action_is_affordable(environment, region, action_name):
    action_id = ACTION_NAMES.index(action_name)
    development_choice, investment_choice = action_parts(action_id)
    cost = environment.config.development_cost if development_choice else 0.0

    if investment_choice in (1, 2, 3):
        cost += environment.config.prevention_costs[investment_choice]
    elif investment_choice in (4, 5, 6):
        cost += environment.config.cleanup_costs[investment_choice - 3]

    can_develop = (
        not development_choice
        or environment.development_indices[region] < 100
    )
    return can_develop and environment.currency_balances[region] >= cost

def choose_rule_action(environment, region, policy):
    edi = environment.development_indices[region]
    developing = edi < 100
    pressure = 0.0

    for pollutant in environment.config.pollutants:
        load = environment.pollutant_loads[pollutant][region]
        limit = environment.config.pollution_limits[pollutant]
        pressure = max(pressure, load / limit)

    if policy == "sequential_high_prevention":
        preferred = "maintain_prevent_high"
        fallback = ["maintain_prevent_high", "maintain_clean_high", "maintain_prevent_medium", "maintain_none"]

        if developing:
            preferred = "develop_none"
            fallback = ["develop_none", "develop_prevent_medium", "maintain_prevent_high", "maintain_none"]

    elif policy == "develop_medium_then_high":
        preferred = "maintain_prevent_high"
        fallback = ["maintain_prevent_high", "maintain_clean_high", "maintain_prevent_medium", "maintain_none"]

        if developing:
            preferred = "develop_prevent_medium"
            fallback = ["develop_prevent_medium", "develop_none", "develop_prevent_high", "maintain_prevent_high", "maintain_none"]

    elif policy == "threshold_adaptive":
        if pressure >= 0.45:
            preferred = "maintain_clean_high"
            fallback = ["maintain_clean_high", "maintain_prevent_high", "maintain_clean_medium", "maintain_none"]

            if developing:
                preferred = "develop_clean_high"
                fallback = ["develop_clean_high", "develop_prevent_high", "develop_prevent_medium", "develop_none", "maintain_clean_high", "maintain_prevent_high", "maintain_none"]

        elif pressure >= 0.25:
            preferred = "maintain_prevent_high"
            fallback = ["maintain_prevent_high", "maintain_clean_high", "maintain_prevent_medium", "maintain_none"]

            if developing:
                preferred = "develop_prevent_high"
                fallback = ["develop_prevent_high", "develop_prevent_medium", "develop_none", "maintain_prevent_high", "maintain_none"]

        else:
            preferred = "maintain_prevent_medium"
            fallback = ["maintain_prevent_medium", "maintain_prevent_low", "maintain_none"]

            if developing:
                preferred = "develop_prevent_medium"
                fallback = ["develop_prevent_medium", "develop_none", "develop_prevent_low", "maintain_prevent_medium", "maintain_none"]

    else:
        raise ValueError(f"Unknown fixed policy: {policy}")

    action_candidates = [preferred]

    for action_name in fallback:
        if action_name not in action_candidates:
            action_candidates.append(action_name)

    for action_name in action_candidates:
        if action_is_affordable(environment, region, action_name):
            return ACTION_NAMES.index(action_name)

    return ACTION_NAMES.index("maintain_none")




def proposed_local_rewards(info, config, pollution_weight=2.0):
    values = []

    for region in config.regions:
        pressure = 0.0

        for pollutant in config.pollutants:
            load = info["pollutant_loads"][pollutant][region]
            limit = config.pollution_limits[pollutant]
            pollutant_pressure = load / limit

            pressure = max(pressure, pollutant_pressure)

        failed = False

        for failed_region, pollutant in info["failed_regions"]:
            if failed_region == region:
                failed = True
                break

        development_cost = info["development_costs_paid"][region]
        prevention_cost = info["prevention_costs_paid"][region]
        cleanup_cost = info["cleanup_costs_paid"][region]
        mitigation_cost = prevention_cost + cleanup_cost

        development_reward = info["development_gains"][region] / 10.0
        development_reward += 0.75 * info["development_indices"][region] / 100.0

        income_reward = (info["income"][region] - config.income_base) / 100.0

        pollution_penalty = pollution_weight * info["pollution_damage"][region]
        risk_penalty = 100.0 * max(0.0, pressure - 0.25) ** 2
        development_penalty = 0.05 * development_cost / 100.0
        mitigation_penalty = 0.50 * mitigation_cost / 100.0
        failure_penalty = 10.0 * failed
        invalid_penalty = 5.0 * info["invalid_attempts"][region]

        value = development_reward + income_reward
        value -= pollution_penalty
        value -= risk_penalty
        value -= development_penalty
        value -= mitigation_penalty
        value -= failure_penalty
        value -= invalid_penalty

        values.append(value)

    return values



def _score_action(environment, region, action_id, reward_name):
    trial = deepcopy(environment)
    actions = [ACTION_NAMES.index("maintain_prevent_medium")] * len(trial.config.regions)
    actions[region] = action_id
    _, _, terminated, _, info = trial.step(actions)

    if reward_name == "original_reward_greedy":
        score = info["mixed_rewards"][region]
    else:
        local_values = proposed_local_rewards(info, trial.config)
        score = 0.7 * local_values[region] + 0.3 * np.mean(local_values)

    if terminated:
        score -= trial.config.failure_penalty

    return score


def _greedy_actions(environment, reward_name):
    actions = []

    for region in environment.config.regions:
        scores = []

        for action_id in range(len(ACTION_NAMES)):
            score = _score_action(environment, region, action_id, reward_name)
            scores.append(score)

        best_action = int(np.argmax(scores))
        actions.append(best_action)

    return actions

def _episode_actions(environment, policy):
    if policy in ("original_reward_greedy", "proposed_reward_greedy"):
        return _greedy_actions(environment, policy)

    actions = []

    for region in environment.config.regions:
        action = choose_rule_action(environment, region, policy)
        actions.append(action)

    return actions


def run_policy_episode(policy, seed, starting_cash, initial_pollution):
    config = EnvironmentConfig()

    config.initial_pollution = {}

    for pollutant in config.pollutants:
        config.initial_pollution[pollutant] = initial_pollution

    environment = EconomicSustainabilityEnvironment(reward_mode="local", config=config, reward_formula="original")
    environment.reset(seed)

    offsets = []
    balances = []

    for region in config.regions:
        offset = CASH_OFFSETS[(region + seed) % len(CASH_OFFSETS)]
        offsets.append(offset)
        balances.append(starting_cash + offset)

    environment.currency_balances = tuple(balances)

    environment.pollutant_loads = {}

    for pollutant in config.pollutants:
        environment.pollutant_loads[pollutant] = (initial_pollution,) * len(config.regions)

    environment._update_indices()

    mean_gpi_total = 0.0
    peak_gpi = environment.global_pollution_index
    peak_pressure = 0.0
    mitigation_spending = 0.0
    development_spending = 0.0
    original_return = 0.0
    proposed_return = 0.0
    threshold_region_steps = 0
    region_steps = 0

    failed = False
    failure_step = np.nan
    failure_pollutants = set()
    target_reached_step = np.nan

    for step in range(config.episode_steps):
        actions = _episode_actions(environment, policy)
        _, _, terminated, truncated, info = environment.step(actions)

        local_original = np.asarray(info["local_rewards"], dtype=float)
        local_proposed = np.asarray(proposed_local_rewards(info, config), dtype=float)

        if terminated:
            local_original -= config.failure_penalty
            local_proposed -= config.failure_penalty

        mixed_original = 0.7 * local_original + 0.3 * np.mean(local_original)
        mixed_proposed = 0.7 * local_proposed + 0.3 * np.mean(local_proposed)

        original_return += np.mean(mixed_original)
        proposed_return += np.mean(mixed_proposed)

        prevention_spending = sum(info["prevention_costs_paid"])
        cleanup_spending = sum(info["cleanup_costs_paid"])

        mitigation_spending += prevention_spending + cleanup_spending
        development_spending += sum(info["development_costs_paid"])

        mean_gpi_total += environment.global_pollution_index
        peak_gpi = max(peak_gpi, environment.global_pollution_index)

        for pollutant in config.pollutants:
            limit = config.pollution_limits[pollutant]

            for region in config.regions:
                load = environment.pollutant_loads[pollutant][region]
                pressure = load / limit

                peak_pressure = max(peak_pressure, pressure)

                if pressure >= 0.50:
                    threshold_region_steps += 1

                region_steps += 1

        if min(environment.development_indices) >= 100 and np.isnan(target_reached_step):
            target_reached_step = environment.step_count

        if terminated:
            failed = True
            failure_step = environment.step_count

            for region, pollutant in info["failed_regions"]:
                failure_pollutants.add(pollutant)

            break

        if truncated:
            break

    steps = environment.step_count
    final_edi_mean = np.mean(environment.development_indices)
    final_edi_min = min(environment.development_indices)
    success = not failed and final_edi_min >= 100

    results = {
        "policy": policy,
        "seed": seed,
        "starting_cash": starting_cash,
        "initial_pollution": initial_pollution,
        "success_EDI_100": success,
        "survived": not failed,
        "steps": steps,
        "target_reached_step": target_reached_step,
        "failure_step": failure_step,
        "failure_pollutants": ", ".join(sorted(failure_pollutants)),
        "final_EDI_mean": final_edi_mean,
        "final_EDI_min": final_edi_min,
        "mean_GPI": mean_gpi_total / steps,
        "peak_GPI": peak_gpi,
        "peak_pressure": peak_pressure,
        "threshold_exceedance_rate": threshold_region_steps / max(region_steps, 1),
        "mitigation_spending": mitigation_spending,
        "development_spending": development_spending,
        "original_mixed_return_per_step": original_return / steps,
        "proposed_mixed_return_per_step": proposed_return / steps,
    }

    return results


def run_policy_comparison():
    rows = []
    for starting_cash in COMPARISON_CASH:
        for initial_pollution in COMPARISON_POLLUTION:
            for seed in COMPARISON_SEEDS:
                for policy in COMPARISON_POLICIES:
                    result = run_policy_episode(policy, seed, starting_cash, initial_pollution)
                    rows.append(result)

    results = pd.DataFrame(rows)

    summary = results.groupby(["policy", "starting_cash", "initial_pollution"]).agg(
        runs=("seed", "count"),
        success_rate=("success_EDI_100", "mean"),
        survival_rate=("survived", "mean"),
        final_EDI=("final_EDI_mean", "mean"),
        final_EDI_min=("final_EDI_min", "mean"),
        mean_GPI=("mean_GPI", "mean"),
        peak_GPI=("peak_GPI", "mean"),
        peak_pressure=("peak_pressure", "mean"),
        threshold_exceedance=("threshold_exceedance_rate", "mean"),
        mitigation_spending=("mitigation_spending", "mean"),
        development_spending=("development_spending", "mean"),
        original_reward=("original_mixed_return_per_step", "mean"),
        proposed_reward=("proposed_mixed_return_per_step", "mean"),
    ).reset_index()

    return results, summary




