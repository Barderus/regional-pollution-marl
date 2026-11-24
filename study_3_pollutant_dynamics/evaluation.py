"""Episode evaluation helpers for Study 3 pollutant dynamics."""


def run_fixed_policy_episode(environment, policy, seed):
    observations = environment.reset(seed)
    trajectory = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        actions = policy(observations, environment)
        next_observations, terminated, truncated, info = environment.step(actions)
        trajectory.append(info)
        observations = next_observations

    return trajectory, terminated, truncated


def summarize_trajectory(trajectory, config, terminated, truncated):
    final_info = trajectory[-1]
    summary = {
        "steps": len(trajectory),
        "failure": terminated,
        "completed_horizon": truncated,
        "failed_pollutants": final_info["failed_pollutants"],
    }

    for pollutant in config.pollutants:
        stock_history = []
        mobile_stock_history = []

        for step_info in trajectory:
            stock = step_info["stocks"][pollutant]
            mobile_stock = step_info["mobile_stocks"].get(pollutant, stock)

            if pollutant == config.global_pollutant:
                stock_history.append(stock)
                mobile_stock_history.append(stock)
            else:
                stock_history.extend(stock)
                mobile_stock_history.extend(mobile_stock)

        summary[f"mean_{pollutant}_stock"] = sum(stock_history) / len(stock_history)
        summary[f"peak_{pollutant}_stock"] = max(stock_history)
        summary[f"final_{pollutant}_stock"] = final_info["stocks"][pollutant]
        summary[f"mean_{pollutant}_mobile_stock"] = (
            sum(mobile_stock_history) / len(mobile_stock_history)
        )
        summary[f"peak_{pollutant}_mobile_stock"] = max(mobile_stock_history)
        summary[f"final_{pollutant}_mobile_stock"] = final_info[
            "mobile_stocks"
        ].get(pollutant, final_info["stocks"][pollutant])

    return summary
