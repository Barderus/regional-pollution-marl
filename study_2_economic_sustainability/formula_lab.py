"""Notebook helpers for testing Study 2 equation candidates without RL."""
import random
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

INVESTMENT_NAMES = (
    "none",
    "prevent_low",
    "prevent_medium",
    "prevent_high",
    "clean_low",
    "clean_medium",
    "clean_high",
)


def make_scenario(**changes):
    """Return source-matched defaults with selected equation changes."""
    scenario = {
        "emission_shape": "linear",
        "income_shape": "linear",
        "development_shape": "current",
        "emission_multiplier": 1.0,
        "income_multiplier": 1.0,
        "prevention_multiplier": 1.0,
        "prevention_cost_multiplier": 1.0,
        "cleanup_capacity_multiplier": 1.0,
        "cleanup_cost_multiplier": 1.0,
        "recovery_multiplier": 1.0,
    }
    scenario.update(changes)
    return scenario


def development_curve(edi, shape, exponential_rate=2.0):
    """Scale an EDI value from zero to one using a candidate curve."""
    scaled_edi = np.clip(np.asarray(edi, dtype=float) / 100.0, 0.0, 1.0)

    if shape == "linear":
        return scaled_edi
    if shape == "concave":
        return np.sqrt(scaled_edi)
    if shape == "quadratic":
        return scaled_edi ** 2
    if shape == "exponential":
        return np.expm1(exponential_rate * scaled_edi) / np.expm1(exponential_rate)

    raise ValueError(f"Unknown curve shape: {shape}")


def calculate_emissions(config, edi, pollutant, shape="linear", multiplier=1.0):
    development_emissions = config.edi_emissions_per_point[pollutant] * 100
    development_emissions *= development_curve(edi, shape)
    development_emissions *= multiplier

    return config.baseline_emissions[pollutant] + development_emissions


def calculate_income(config, edi, shape="linear", multiplier=1.0):
    development_income = config.income_per_edi * 100
    development_income *= development_curve(edi, shape)
    development_income *= multiplier

    return config.income_base + development_income


def calculate_development_gain(edi, shape):
    if shape == "current":
        if edi < 40:
            return 10
        if edi < 70:
            return 5
        if edi < 100:
            return 2
        return 0

    if shape == "linear":
        return 10 if edi < 100 else 0

    if shape == "quadratic":
        if edi >= 100:
            return 0
        return max(1, round(10 * (1 - (edi / 100) ** 2)))

    raise ValueError(f"Unknown development shape: {shape}")


def get_investment(config, scenario, level):
    if level <= 3:
        cost = config.prevention_costs[level]
        cost *= scenario["prevention_cost_multiplier"]
        prevention = config.prevention_fractions[level]
        prevention *= scenario["prevention_multiplier"]
        return cost, min(1.0, prevention), 0.0

    cleanup_level = level - 3
    cost = config.cleanup_costs[cleanup_level]
    cost *= scenario["cleanup_cost_multiplier"]
    capacity = config.cleanup_capacities[cleanup_level]
    capacity *= scenario["cleanup_capacity_multiplier"]

    return cost, 0.0, capacity


def choose_action_details(config, policy_name, region, development, currency,
                          pollution, scenario, development_target):
    pressure = 0.0

    for pollutant in config.pollutants:
        load = pollution[pollutant][region]
        limit = config.pollution_limits[pollutant]
        pollutant_pressure = load / limit

        pressure = max(pressure, pollutant_pressure)

    should_develop = development[region] < development_target

    if policy_name == "growth_with_prevention":
        preferred = 3
    elif policy_name == "staged_adaptive":
        preferred = 6 if pressure >= 0.65 else 3
    elif policy_name == "resource_aware":
        preferred = 6 if pressure >= 0.50 else 3
    elif policy_name == "growth_only_diagnostic":
        preferred = 0
    else:
        raise ValueError(f"Unknown policy: {policy_name}")

    preferred_cost, _, _ = get_investment(config, scenario, preferred)
    preferred_development_cost = (config.development_cost if should_develop else 0.0)
    preferred_action_affordable = (currency[region] >= preferred_cost + preferred_development_cost)

    investment_order = list(dict.fromkeys([preferred, 2, 1, 0, 5, 4, 6]))
    for level in investment_order:
        investment_cost, _, _ = get_investment(config, scenario, level)
        development_cost = config.development_cost if should_develop else 0.0
        if currency[region] >= investment_cost + development_cost:
            return {
                "develop": should_develop,
                "investment_level": level,
                "preferred_investment": preferred,
                "preferred_action_affordable": preferred_action_affordable,
                "affordability_limited": level != preferred,
            }

    if should_develop and currency[region] >= config.development_cost:
        return {
            "develop": True,
            "investment_level": 0,
            "preferred_investment": preferred,
            "preferred_action_affordable": preferred_action_affordable,
            "affordability_limited": preferred != 0,
        }

    return {
        "develop": False,
        "investment_level": 0,
        "preferred_investment": preferred,
        "preferred_action_affordable": preferred_action_affordable,
        "affordability_limited": True,
    }


def choose_action(config, policy_name, region, development, currency, pollution,
                  scenario, development_target):
    """Return the policy's chosen development and investment decisions."""
    details = choose_action_details(
        config, policy_name, region, development, currency, pollution,
        scenario, development_target,
    )
    return details["develop"], details["investment_level"]


def run_episode(config, policy_name, seed, starting_currency, scenario,
                development_target=60, record_trace=False):
    """Run one deterministic fixed-policy episode in the equation sandbox."""
    rng = random.Random(seed)
    regions = config.regions
    pollutants = config.pollutants
    development = [0.0] * len(regions)

    minimum_cash = max(config.initial_currency_minimum, starting_currency - 100)
    maximum_cash = min(config.initial_currency_maximum, starting_currency + 100)
    currency = [float(rng.randint(minimum_cash, maximum_cash)) for _ in regions]
    pollution = {pollutant: [float(config.initial_pollution[pollutant]) for _ in regions] for pollutant in pollutants}

    income_total = 0.0
    gross_emissions_total = 0.0
    prevention_spending = 0.0
    cleanup_spending = 0.0
    invalid_region_steps = 0
    affordability_limited_actions = 0
    capacity_shortfall_region_steps = 0
    prevented_emissions_total = 0.0
    cleanup_removed_total = 0.0
    natural_removal_total = 0.0
    peak_pollutant_load = max(max(pollution[pollutant]) for pollutant in pollutants)
    action_counts = Counter()
    trajectory = []
    failed = False
    failure_step = np.nan
    failure_affordability_limited_actions = 0
    failure_capacity_shortfalls = 0

    for step in range(1, config.episode_steps + 1):
        development_before = list(development)
        currency_before = list(currency)
        pollution_before = {pollutant: list(pollution[pollutant]) for pollutant in pollutants}
        action_details = []
        for region in regions:
            details = choose_action_details(
                config, policy_name, region, development, currency, pollution,
                scenario, development_target,
            )
            action_details.append(details)

        next_development = []
        next_currency = []
        valid_actions = []
        investment_levels = []
        step_income = []
        step_cost = []
        step_development_cost = []

        for region, details in enumerate(action_details):
            develop = details["develop"]
            level = details["investment_level"]
            investment_cost, _, _ = get_investment(config, scenario, level)
            development_cost = config.development_cost if develop else 0.0
            total_cost = investment_cost + development_cost
            valid = currency[region] >= total_cost
            valid = valid and (not develop or development[region] < 100)

            gain = 0.0

            if valid and develop:
                gain = calculate_development_gain(development[region], scenario["development_shape"])
                gain = min(100 - development[region], gain)

            current_income = calculate_income(config, development[region], scenario["income_shape"],
                                              scenario["income_multiplier"])
            paid_cost = 0.0
            selected_level = 0
            paid_investment_cost = 0.0
            paid_development_cost = 0.0

            if valid:
                paid_cost = total_cost
                selected_level = level
                paid_investment_cost = investment_cost
                paid_development_cost = development_cost

            next_development.append(development[region] + gain)
            next_currency.append(currency[region] - paid_cost + current_income)
            valid_actions.append(valid)
            investment_levels.append(selected_level)
            step_income.append(current_income)
            step_cost.append(paid_investment_cost)
            step_development_cost.append(paid_development_cost)

            action_name = "develop" if develop else "maintain"
            action_name += "_" + INVESTMENT_NAMES[level]
            action_counts[action_name] += 1
            if details["affordability_limited"]:
                affordability_limited_actions += 1

        gross_emissions = {pollutant: [] for pollutant in pollutants}
        remaining_emissions = {pollutant: [] for pollutant in pollutants}
        local_emissions = {pollutant: [] for pollutant in pollutants}
        prevented_emissions = {pollutant: [] for pollutant in pollutants}

        for pollutant in pollutants:
            for region in regions:
                amount = calculate_emissions(config, development[region], pollutant, scenario["emission_shape"],
                                             scenario["emission_multiplier"])
                _, prevention_fraction, _ = get_investment(config, scenario, investment_levels[region])
                gross_emissions[pollutant].append(amount)
                prevented = amount * prevention_fraction
                prevented_emissions[pollutant].append(prevented)
                remaining_emissions[pollutant].append(amount - prevented)

            transport = config.transport_fractions[pollutant]
            for region in regions:
                incoming = 0.0
                for neighbor in config.neighbors[region]:
                    incoming += (remaining_emissions[pollutant][neighbor] * transport / len(config.neighbors[neighbor])                    )
                retained = remaining_emissions[pollutant][region]
                local_emissions[pollutant].append(retained * (1 - transport) + incoming)

        next_pollution = {pollutant: [] for pollutant in pollutants}
        natural_removal = {pollutant: [] for pollutant in pollutants}
        incoming_pollution = {pollutant: [] for pollutant in pollutants}
        load_before_cleanup = {pollutant: [] for pollutant in pollutants}
        cleanup_removed = {pollutant: [] for pollutant in pollutants}
        cleanup_capacities = {pollutant: [] for pollutant in pollutants}
        cleanup_efficiencies = {pollutant: [] for pollutant in pollutants}
        for pollutant in pollutants:
            transport = config.transport_fractions[pollutant]
            limit = config.pollution_limits[pollutant]
            recovered_stock = []

            for region in regions:
                severity = min(pollution[pollutant][region] / limit, 1.0)
                recovery_factor = 1 - config.recovery_degradation[pollutant] * severity
                recovery_factor = max(config.minimum_recovery_fraction, recovery_factor)
                removal_fraction = config.natural_recovery_rates[pollutant]
                removal_fraction *= scenario["recovery_multiplier"] * recovery_factor
                removal_fraction = min(1.0, removal_fraction)
                removed = pollution[pollutant][region] * removal_fraction
                natural_removal[pollutant].append(removed)
                recovered_stock.append(pollution[pollutant][region] - removed)

            for region in regions:
                incoming = 0.0
                for neighbor in config.neighbors[region]:
                    incoming += (recovered_stock[neighbor] * transport / len(config.neighbors[neighbor]))
                incoming_pollution[pollutant].append(incoming)
                load = recovered_stock[region] * (1 - transport) + incoming
                load += local_emissions[pollutant][region]
                load_before_cleanup[pollutant].append(load)

                _, _, capacity = get_investment(config, scenario, investment_levels[region])
                severity = min(load / limit, 1.0)
                efficiency = 1 - config.cleanup_difficulty[pollutant] * severity
                efficiency = max(config.minimum_cleanup_efficiency, efficiency)
                cleanup = min(load, capacity * efficiency)
                cleanup_removed[pollutant].append(cleanup)
                cleanup_capacities[pollutant].append(capacity)
                cleanup_efficiencies[pollutant].append(efficiency)
                next_pollution[pollutant].append(load - cleanup)

        income_total += sum(step_income)
        gross_emissions_total += sum(sum(gross_emissions[pollutant]) for pollutant in pollutants)
        invalid_region_steps += sum(not valid for valid in valid_actions)
        prevented_emissions_step = sum(sum(prevented_emissions[pollutant]) for pollutant in pollutants)
        cleanup_removed_step = sum(sum(cleanup_removed[pollutant]) for pollutant in pollutants)
        natural_removal_step = sum(sum(natural_removal[pollutant]) for pollutant in pollutants)
        prevented_emissions_total += prevented_emissions_step
        cleanup_removed_total += cleanup_removed_step
        natural_removal_total += natural_removal_step
        for region, level in enumerate(investment_levels):
            if level <= 3:
                prevention_spending += step_cost[region]
            else:
                cleanup_spending += step_cost[region]

        development = next_development
        currency = next_currency
        pollution = next_pollution
        step_peak = max(max(pollution[pollutant]) for pollutant in pollutants)
        peak_pollutant_load = max(peak_pollutant_load, step_peak)

        failed_now = False
        for pollutant in pollutants:
            limit = config.pollution_limits[pollutant]

            for load in pollution[pollutant]:
                if load >= limit:
                    failed_now = True
                    break
            if failed_now:
                break

        capacity_shortfalls = 0
        for region in regions:
            for pollutant in pollutants:
                limit = config.pollution_limits[pollutant]
                current_load = pollution[pollutant][region]
                load_after_cleanup = load_before_cleanup[pollutant][region] - cleanup_removed[pollutant][region]

                if current_load >= limit and investment_levels[region] > 3 and load_after_cleanup >= limit:
                    capacity_shortfalls += 1

        capacity_shortfall_region_steps += capacity_shortfalls

        if failed_now:
            failure_affordability_limited_actions = 0
            for details in action_details:
                failure_affordability_limited_actions += details["affordability_limited"]

            failure_capacity_shortfalls = capacity_shortfalls

        if record_trace:
            for region, details in enumerate(action_details):
                gross_total = 0.0
                prevented_total = 0.0
                cleanup_total = 0.0
                natural_removal_total = 0.0
                weighted_pollution = 0.0

                for pollutant in pollutants:
                    gross_total += gross_emissions[pollutant][region]
                    prevented_total += prevented_emissions[pollutant][region]
                    cleanup_total += cleanup_removed[pollutant][region]
                    natural_removal_total += natural_removal[pollutant][region]

                    load = pollution[pollutant][region]
                    limit = config.pollution_limits[pollutant]
                    normalized_load = min(load / limit, 1.0)

                    weighted_pollution += config.pollutant_weights[pollutant] * normalized_load

                regional_gpi = config.gpi_scale * weighted_pollution
                emissions_prevented_fraction = prevented_total / max(gross_total, 1e-12)

                mitigation_cost_to_income = np.nan

                if step_income[region]:
                    mitigation_cost_to_income = step_cost[region] / step_income[region]

                investment = investment_levels[region]
                prevention_effectiveness = get_investment(config, scenario, investment)[1]

                trace_row = {
                    "step": step,
                    "region": region,
                    "edi_before": development_before[region],
                    "edi_after": development[region],
                    "currency_before": currency_before[region],
                    "currency_after": currency[region],
                    "income": step_income[region],
                    "development_cost": step_development_cost[region],
                    "mitigation_cost": step_cost[region],
                    "mitigation_cost_to_income": mitigation_cost_to_income,
                    "preferred_investment": INVESTMENT_NAMES[details["preferred_investment"]],
                    "selected_investment": INVESTMENT_NAMES[investment],
                    "preferred_action_affordable": details["preferred_action_affordable"],
                    "affordability_limited": details["affordability_limited"],
                    "regional_gpi": regional_gpi,
                    "gross_emissions_total": gross_total,
                    "prevention_effectiveness": prevention_effectiveness,
                    "emissions_prevented_fraction": emissions_prevented_fraction,
                    "emissions_prevented_total": prevented_total,
                    "cleanup_removed_total": cleanup_total,
                    "natural_removal_total": natural_removal_total,
                    "failed_this_step": failed_now,
                }

                for pollutant in pollutants:
                    trace_row[f"pollution_before_{pollutant}"] = pollution_before[pollutant][region]
                    trace_row[f"gross_emissions_{pollutant}"] = gross_emissions[pollutant][region]
                    trace_row[f"emissions_prevented_{pollutant}"] = prevented_emissions[pollutant][region]
                    trace_row[f"local_emissions_{pollutant}"] = local_emissions[pollutant][region]
                    trace_row[f"natural_removal_{pollutant}"] = natural_removal[pollutant][region]
                    trace_row[f"incoming_pollution_{pollutant}"] = incoming_pollution[pollutant][region]
                    trace_row[f"load_before_cleanup_{pollutant}"] = load_before_cleanup[pollutant][region]
                    trace_row[f"cleanup_capacity_{pollutant}"] = cleanup_capacities[pollutant][region]
                    trace_row[f"cleanup_efficiency_{pollutant}"] = cleanup_efficiencies[pollutant][region]
                    trace_row[f"cleanup_removed_{pollutant}"] = cleanup_removed[pollutant][region]
                    trace_row[f"pollution_after_{pollutant}"] = pollution[pollutant][region]

                trajectory.append(trace_row)

        if failed_now:
            failed = True
            failure_step = step
            break

    final_pollution = {}

    for pollutant in pollutants:
        final_pollution[pollutant] = float(np.mean(pollution[pollutant]))

    final_peak_pollutant_load = max(max(pollution[pollutant]) for pollutant in pollutants)

    # These flags describe constraints observed on the failure step
    if not failed:
        failure_mechanism = "no_failure"
    elif failure_affordability_limited_actions and failure_capacity_shortfalls:
        failure_mechanism = "unaffordable_preference_and_cleanup_capacity_insufficient"
    elif failure_affordability_limited_actions:
        failure_mechanism = "preferred_action_unaffordable"
    elif failure_capacity_shortfalls:
        failure_mechanism = "cleanup_capacity_insufficient"
    else:
        failure_mechanism = "failure_without_affordability_or_cleanup_capacity_flag"

    final_mean_edi = float(np.mean(development))
    final_mean_currency = float(np.mean(currency))
    mitigation_spending = prevention_spending + cleanup_spending

    success = not failed and final_mean_edi >= development_target
    failure_affordability_constrained = failure_affordability_limited_actions > 0
    failure_cleanup_capacity_limited = failure_capacity_shortfalls > 0

    results = {
        "failed": failed,
        "survived": not failed,
        "success": success,
        "failure_step": failure_step,
        "steps": step,
        "final_mean_edi": final_mean_edi,
        "final_mean_currency": final_mean_currency,
        "income_total": income_total,
        "prevention_spending": prevention_spending,
        "cleanup_spending": cleanup_spending,
        "mitigation_spending": mitigation_spending,
        "gross_emissions": gross_emissions_total,
        "emissions_prevented": prevented_emissions_total,
        "cleanup_removed": cleanup_removed_total,
        "natural_removal": natural_removal_total,
        "peak_pollutant_load": peak_pollutant_load,
        "final_peak_pollutant_load": final_peak_pollutant_load,
        "invalid_region_steps": invalid_region_steps,
        "affordability_limited_actions": affordability_limited_actions,
        "capacity_shortfall_region_steps": capacity_shortfall_region_steps,
        "failure_affordability_limited_actions": failure_affordability_limited_actions,
        "failure_capacity_shortfalls": failure_capacity_shortfalls,
        "failure_affordability_constrained": failure_affordability_constrained,
        "failure_cleanup_capacity_limited": failure_cleanup_capacity_limited,
        "failure_mechanism": failure_mechanism,
        "action_counts": dict(action_counts),
    }

    for pollutant in pollutants:
        results[f"final_{pollutant}"] = final_pollution[pollutant]

    if record_trace:
        results["trajectory"] = trajectory

    return results


def evaluate_scenario(config, name, scenario, seeds, cash_levels, development_target=60, record_traces=False):
    policies = ("growth_with_prevention", "staged_adaptive", "resource_aware")
    rows = []

    for policy_name, seed, starting_currency in product(policies, seeds, cash_levels):
        episode = run_episode(config, policy_name, seed, starting_currency, scenario, development_target,
                              record_trace=record_traces)
        rows.append({
            "scenario": name,
            "policy": policy_name,
            "seed": seed,
            "starting_currency_tier": starting_currency,
            **episode,
        })

    return pd.DataFrame(rows)



def summarize_results(results):
    summary = results.groupby(["scenario", "policy"]).agg(
        scenario_runs=("success", "size"),
        success_rate=("success", "mean"),
        survival_rate=("survived", "mean"),
        failure_rate=("failed", "mean"),
        mean_final_edi=("final_mean_edi", "mean"),
        mean_final_currency=("final_mean_currency", "mean"),
        mean_peak_pollution=("peak_pollutant_load", "mean"),
        mean_final_peak_pollution=("final_peak_pollutant_load", "mean"),
        mean_mitigation_spending=("mitigation_spending", "mean"),
        mean_income=("income_total", "mean"),
        mean_gross_emissions=("gross_emissions", "mean"),
        mean_emissions_prevented=("emissions_prevented", "mean"),
        mean_cleanup_removed=("cleanup_removed", "mean"),
        mean_affordability_limited_actions=("affordability_limited_actions", "mean"),
        mean_capacity_shortfall_region_steps=("capacity_shortfall_region_steps", "mean"),
        affordability_constraint_observed_rate=("failure_affordability_constrained", "mean"),
        cleanup_capacity_insufficient_rate=("failure_cleanup_capacity_limited", "mean"),
        mean_failure_step=("failure_step", "mean"),
    ).reset_index()

    return summary

