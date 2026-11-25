"""Shared fixed baseline values for Study 3 notebook consistency checks."""

REFERENCE_BASELINE_SETTINGS = {"episode_steps": 40, "seed": 0, "starting_currency_multiplier": 1.0,
                               "development_cost_shape": "constant", "development_cost_multiplier": 1.0,
                               "investment_cost_multiplier": 1.0, "income_damage_multiplier": 1.0,
                               "income_damage_persistence": 1.0}

REFERENCE_BASELINE_POLICIES = ("development_only", "balanced", "cleanup_focused")

REFERENCE_BASELINE_METRICS = {
    "development_only": {
        "final_mean_EDI": 98.75,
        "final_min_EDI": 95.0,
        "final_mean_currency": 3744.5074013845606,
        "minimum_currency": 475.60531070865056,
        "development_spending": 3375.0,
        "mitigation_spending": 0.0,
        "investment_package_purchases": 0,
        "unaffordable_development_attempts": 0,
        "final_CO2_ppm": 529.6620551647025,
        "final_mean_PM2.5_ug_m3": 8.6135197711126,
        "final_mean_nitrate_mg_N_L": 3.209427191464771,
        "final_mean_lead_total_kg": 17.122550600118057,
    },
    "balanced": {
        "final_mean_EDI": 98.75,
        "final_min_EDI": 95.0,
        "final_mean_currency": 2735.4036443790633,
        "minimum_currency": 275.60531070865056,
        "development_spending": 3375.0,
        "mitigation_spending": 6400.0,
        "investment_package_purchases": 32,
        "unaffordable_development_attempts": 0,
        "final_CO2_ppm": 526.6103046795712,
        "final_mean_PM2.5_ug_m3": 8.570139828334451,
        "final_mean_nitrate_mg_N_L": 2.4450407133818626,
        "final_mean_lead_total_kg": 11.707335869242954,
    },
    "cleanup_focused": {
        "final_mean_EDI": 98.75,
        "final_min_EDI": 95.0,
        "final_mean_currency": 3188.0629772687034,
        "minimum_currency": 375.60531070865056,
        "development_spending": 3375.0,
        "mitigation_spending": 3200.0,
        "investment_package_purchases": 32,
        "unaffordable_development_attempts": 0,
        "final_CO2_ppm": 529.5229404666715,
        "final_mean_PM2.5_ug_m3": 8.6135197711126,
        "final_mean_nitrate_mg_N_L": 3.0869261516748168,
        "final_mean_lead_total_kg": 12.907787368550377,
    },
}


def compare_reference_baseline(result_rows, policy_names=None, tolerance=1e-6):
    if policy_names is None:
        policy_names = REFERENCE_BASELINE_POLICIES

    check_rows = []
    for policy_name in policy_names:
        matching_rows = []
        for result_row in result_rows:
            if result_row["policy"] == policy_name:
                matching_rows.append(result_row)

        unique_result_found = len(matching_rows) == 1
        current_row = None
        if unique_result_found:
            current_row = matching_rows[0]

        for metric_name, expected_value in REFERENCE_BASELINE_METRICS[policy_name].items():
            actual_value = None
            metric_passed = False
            if current_row is not None and metric_name in current_row:
                actual_value = current_row[metric_name]
                metric_passed = abs(float(actual_value) - expected_value) <= tolerance

            check_rows.append({
                "policy": policy_name,
                "metric": metric_name,
                "expected": expected_value,
                "actual": actual_value,
                "passed": unique_result_found and metric_passed,
            })

    return check_rows
