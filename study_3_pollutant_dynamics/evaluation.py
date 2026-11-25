"""Episode summaries for pollutant-only Study 3 simulations."""

def run_fixed_policy_episode(environment, policy, seed):
    observations = environment.reset(seed)
    trajectory = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        actions = policy(observations, environment)
        observations, terminated, truncated, info = environment.step(actions)
        trajectory.append(info)

    return trajectory, terminated, truncated


def summarize_trajectory(trajectory, config, terminated, truncated):
    if not trajectory:
        raise ValueError("Cannot summarize an empty trajectory")

    final_info = trajectory[-1]
    summary = {"steps": len(trajectory), "terminated": terminated, "completed_horizon": truncated,
               "failed_pollutants": final_info["failed_pollutants"]}

    for pollutant in config.pollutants:
        values = []
        for step_info in trajectory:
            stock = step_info["stocks"][pollutant]
            if isinstance(stock, tuple):
                values.extend(stock)
            else:
                values.append(stock)

        if pollutant == "PM2.5":
            summary["mean_PM2.5_exposure_ug_m3"] = sum(values) / len(values)
            summary["peak_PM2.5_exposure_ug_m3"] = max(values)
            summary["final_PM2.5_exposure_ug_m3"] = final_info["stocks"][pollutant]
        elif pollutant == "CO2":
            summary["mean_CO2_ppm"] = sum(values) / len(values)
            summary["peak_CO2_ppm"] = max(values)
            summary["final_CO2_ppm"] = final_info["stocks"][pollutant]
        elif pollutant == "Nitrate-N":
            summary["mean_Nitrate-N_stock_kg"] = sum(values) / len(values)
            summary["peak_Nitrate-N_stock_kg"] = max(values)
            final_stocks = final_info["stocks"][pollutant]
            summary["final_Nitrate-N_stock_kg"] = sum(final_stocks) / len(final_stocks)
            concentrations = []
            for stock in final_stocks:
                concentrations.append(1000 * stock / config.nitrate["water_volume_m3"])
            summary["final_Nitrate-N_mean_mg_L"] = sum(concentrations) / len(concentrations)
        elif pollutant == "Lead":
            summary["mean_Lead_total_stock_kg"] = sum(values) / len(values)
            summary["peak_Lead_total_stock_kg"] = max(values)
            final_stocks = final_info["stocks"][pollutant]
            summary["final_Lead_total_stock_kg"] = sum(final_stocks) / len(final_stocks)
            mobile_values = []
            for step_info in trajectory:
                mobile_values.extend(step_info["mobile_stocks"][pollutant])
            summary["mean_Lead_mobile_stock_kg"] = sum(mobile_values) / len(mobile_values)
            final_mobile = final_info["mobile_stocks"][pollutant]
            summary["final_Lead_mobile_stock_kg"] = sum(final_mobile) / len(final_mobile)

    return summary
