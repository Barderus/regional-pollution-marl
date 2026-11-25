"""Deterministic fixed policies for integrated Study 3 baseline checks."""

import math

from study_3_pollutant_dynamics.integrated_environment import (INTEGRATED_ACTIONS, Study3IntegratedEnvironment)
from study_3_pollutant_dynamics.mitigation_parameters import MITIGATION_POLICIES


FIXED_POLICY_NAMES = ("no_development_no_investment", "development_only", "prevention_focused", "cleanup_focused", "balanced")

def choose_fixed_policy_actions(policy_name, environment):
    if policy_name not in FIXED_POLICY_NAMES:
        raise ValueError(f"Unknown fixed policy: {policy_name}")

    actions = []
    year = environment.step_count + 1
    for region_index in range(len(environment.regions)):
        development_is_active = policy_name != "no_development_no_investment"
        if environment.edi[region_index] >= 100:
            development_is_active = False

        investment_is_due = year >= environment.next_project_purchase_year[region_index]
        investment_name = None
        if investment_is_due and policy_name in MITIGATION_POLICIES:
            if policy_name in ("prevention_focused", "cleanup_focused", "balanced"):
                investment_name = policy_name

        action_name = "maintain"
        if development_is_active:
            action_name = "develop"

        if investment_name is not None:
            if development_is_active:
                action_name = "develop_" + investment_name.replace("_focused", "")
                if investment_name == "balanced":
                    action_name = "develop_balanced"
            else:
                action_name = investment_name.replace("_focused", "")

        if action_name not in INTEGRATED_ACTIONS:
            raise ValueError(f"Fixed policy selected unknown action: {action_name}")
        actions.append(action_name)

    return tuple(actions)


def run_fixed_policy_episode(policy_name, environment, seed=0):
    observations = environment.reset(seed)
    trajectory = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        actions = choose_fixed_policy_actions(policy_name, environment)
        observations, terminated, truncated, info = environment.step(actions)
        trajectory.append(info)

    return trajectory, terminated, truncated


def summarize_integrated_episode(policy_name, trajectory, environment):
    if not trajectory:
        raise ValueError("Cannot summarize an empty episode")

    final_info = trajectory[-1]
    final_economy = final_info["regional_economy"]
    pollutant_values = {"PM2.5": [], "Nitrate-N": [], "Lead": []}

    for info in trajectory:
        pollutant_values["PM2.5"].extend(info["stocks"]["PM2.5"])
        pollutant_values["Nitrate-N"].extend(info["stocks"]["Nitrate-N"])
        pollutant_values["Lead"].extend(info["stocks"]["Lead"])

    mean_values = {}
    for pollutant, values in pollutant_values.items():
        mean_values[pollutant] = sum(values) / len(values)

    minimum_currency = float("inf")
    maximum_edi = 0.0
    maximum_nitrate_residual = 0.0
    maximum_lead_residual = 0.0
    maximum_co2 = 0.0
    stocks_nonnegative = True
    state_values_finite = True
    development_requested_years = 0
    affordable_development_years = 0
    investment_due_years = 0
    affordable_investment_years = 0
    simultaneous_opportunities = 0
    affordable_simultaneous_years = 0
    investment_package_purchases = 0

    for info in trajectory:
        maximum_co2 = max(maximum_co2, info["stocks"]["CO2"])
        for pollutant in environment.config.pollutants:
            stock_values = info["stocks"][pollutant]
            if isinstance(stock_values, tuple):
                for stock_value in stock_values:
                    if stock_value < -1e-9:
                        stocks_nonnegative = False
                    if not math.isfinite(stock_value):
                        state_values_finite = False
            else:
                if stock_values < -1e-9:
                    stocks_nonnegative = False
                if not math.isfinite(stock_values):
                    state_values_finite = False

        for region_result in info["regional_economy"]:
            minimum_currency = min(minimum_currency, region_result["currency"])
            maximum_edi = max(maximum_edi, region_result["EDI"])
            if region_result["development_requested"]:
                development_requested_years += 1
                if region_result["development_affordable"]:
                    affordable_development_years += 1
            if region_result["investment_due"]:
                investment_due_years += 1
                if region_result["investment_affordable"]:
                    affordable_investment_years += 1
            if region_result["package_purchased"]:
                investment_package_purchases += 1
            if region_result["development_requested"] and region_result["investment_due"]:
                simultaneous_opportunities += 1
                if region_result["simultaneous_affordability"]:
                    affordable_simultaneous_years += 1
            numeric_fields = (region_result["EDI"], region_result["currency"], region_result["gross_income"],
                              region_result["net_income"], region_result["damage_fraction"])
            for value in numeric_fields:
                if not math.isfinite(value):
                    state_values_finite = False

        for residual in info["mass_balance_residuals"]["Nitrate-N_kg"]:
            maximum_nitrate_residual = max(maximum_nitrate_residual, abs(residual))

        for residual in info["mass_balance_residuals"]["Lead_kg"]:
            maximum_lead_residual = max(maximum_lead_residual, abs(residual))

    final_nitrate = []
    final_lead = []
    final_mobile_lead = []
    final_pm25 = list(final_info["stocks"]["PM2.5"])

    for region_index in range(len(environment.regions)):
        nitrate_concentration = (1000 * final_info["stocks"]["Nitrate-N"][region_index] / environment.config.nitrate["water_volume_m3"])
        final_nitrate.append(nitrate_concentration)
        final_lead.append(final_info["stocks"]["Lead"][region_index])
        final_mobile_lead.append(final_info["mobile_stocks"]["Lead"][region_index])

    summary = {
        "policy": policy_name,
        "development_cost_shape": environment.development_cost_shape,
        "income_damage_multiplier": environment.income_damage_multiplier,
        "income_damage_persistence": environment.income_damage_persistence,
        "income_retention_fraction": environment.income_retention_fraction,
        "years": len(trajectory),
        "final_mean_EDI": sum(environment.edi) / len(environment.edi),
        "final_min_EDI": min(environment.edi),
        "final_mean_currency": sum(environment.currency) / len(environment.currency),
        "minimum_currency": minimum_currency,
        "maximum_EDI": maximum_edi,
        "peak_CO2_ppm": maximum_co2,
        "final_CO2_ppm": final_info["stocks"]["CO2"],
        "final_mean_PM2.5_ug_m3": sum(final_pm25) / len(final_pm25),
        "final_mean_nitrate_mg_N_L": sum(final_nitrate) / len(final_nitrate),
        "final_mean_lead_total_kg": sum(final_lead) / len(final_lead),
        "final_mean_lead_mobile_kg": sum(final_mobile_lead) / len(final_mobile_lead),
        "mean_PM2.5_ug_m3": mean_values["PM2.5"],
        "mean_Nitrate-N_stock_kg": mean_values["Nitrate-N"],
        "mean_Lead_total_kg": mean_values["Lead"],
        "development_spending": sum(environment.development_spending),
        "mitigation_spending": sum(environment.investment_spending),
        "unaffordable_development_attempts": sum(environment.unaffordable_development_attempts),
        "unaffordable_investment_attempts": sum(environment.unaffordable_investment_attempts),
        "development_requested_region_years": development_requested_years,
        "affordable_development_region_years": affordable_development_years,
        "investment_due_region_years": investment_due_years,
        "affordable_investment_region_years": affordable_investment_years,
        "simultaneous_investment_opportunities": simultaneous_opportunities,
        "affordable_simultaneous_region_years": affordable_simultaneous_years,
        "investment_package_purchases": investment_package_purchases,
        "maximum_nitrate_mass_balance_residual_kg": maximum_nitrate_residual,
        "maximum_lead_mass_balance_residual_kg": maximum_lead_residual,
        "pollutant_stocks_nonnegative": stocks_nonnegative,
        "state_values_finite": state_values_finite,
    }

    return summary


def evaluate_fixed_policies(config, policy_names=None, currency_multipliers=None, seeds=None, development_cost_multiplier=1.0,
                            investment_cost_multiplier=1.0, development_cost_shape="constant", income_damage_multiplier=1.0,
                            income_damage_persistence=1.0, income_retention_fraction=1.0):

    if policy_names is None:
        policy_names = FIXED_POLICY_NAMES
    if currency_multipliers is None:
        currency_multipliers = (0.5, 1.0, 1.5)
    if seeds is None:
        seeds = (0,)

    rows = []
    for policy_name in policy_names:
        for currency_multiplier in currency_multipliers:
            for seed in seeds:
                environment = Study3IntegratedEnvironment(config, starting_currency_multiplier=currency_multiplier,
                                                          development_cost_multiplier=development_cost_multiplier,
                                                          investment_cost_multiplier=investment_cost_multiplier,
                                                          development_cost_shape=development_cost_shape,
                                                          income_damage_multiplier=income_damage_multiplier,
                                                          income_damage_persistence=income_damage_persistence,
                                                          income_retention_fraction=income_retention_fraction
                                                          )

                trajectory, terminated, truncated = run_fixed_policy_episode(policy_name, environment, seed)
                summary = summarize_integrated_episode(policy_name, trajectory, environment )
                summary["starting_currency_multiplier"] = currency_multiplier
                summary["development_cost_multiplier"] = development_cost_multiplier
                summary["investment_cost_multiplier"] = investment_cost_multiplier
                summary["development_cost_shape"] = development_cost_shape
                summary["seed"] = seed
                summary["terminated"] = terminated
                summary["completed_horizon"] = truncated
                rows.append(summary)

    return rows
