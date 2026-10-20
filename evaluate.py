"""Evaluate saved IQL policies and baselines on common held-out demand seeds."""

import csv
import json
import random
import statistics
from collections import defaultdict

from agents import TabularQLearner
from baselines import fixed_medium_action, random_action, threshold_action
from config import TrainingConfig
from environment import PollutionEnvironment

METRICS = ("mean_regional_pollution", "mitigation_cost", "threshold_exceedance",
           "emitted_units", "regional_pollution_spread", "action_0_fraction",
           "action_1_fraction", "action_2_fraction")


def evaluate_policy(policy, environment_seeds):
    rows = []
    environment = PollutionEnvironment()

    for seed in environment_seeds:
        observation = environment.reset(seed)
        load_totals = [0.0] * 4
        total_cost = 0.0
        total_emissions = 0.0
        exceedances = 0
        action_counts = [0] * 3

        for _ in range(environment.config.episode_steps):
            actions = policy(observation)

            for action in actions:
                action_counts[action] += 1
            observation, _, _, info = environment.step(actions)
            loads = info["pollution_loads"]

            for i, load in enumerate(loads):
                load_totals[i] += load
                if load > 30:
                    exceedances += 1

            total_cost += sum(info["mitigation_costs"])
            total_emissions += sum(info["emissions"])

        steps = environment.config.episode_steps
        regional_means = [total / steps for total in load_totals]
        region_steps = 4 * steps
        mean_regional_pollution = sum(load_totals) / region_steps
        threshold_exceedance = exceedances / region_steps
        regional_pollution_spread = max(regional_means) - min(regional_means)

        episode_row = {
            "environment_seed": seed,
            "mean_regional_pollution": mean_regional_pollution,
            "mitigation_cost": total_cost,
            "threshold_exceedance": threshold_exceedance,
            "emitted_units": total_emissions,
            "regional_pollution_spread": regional_pollution_spread,
        }
        for action in range(3):
            episode_row[f"action_{action}_fraction"] = action_counts[action] / region_steps
        rows.append(episode_row)

    return rows


def summarize_by_training_seed(episode_rows):
    groups = defaultdict(list)
    for row in episode_rows:
        groups[(row["policy"], int(row["training_seed"]))].append(row)

    summaries = []

    for (policy, seed), rows in sorted(groups.items()):
        summary = {"policy": policy, "training_seed": seed, "episodes": len(rows)}
        for metric in METRICS:
            summary[metric] = statistics.mean(row[metric] for row in rows)
        summaries.append(summary)

    return summaries


def write_csv(filename, rows):
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    results_directory = "results/raw"

    with open(f"{results_directory}/manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)

    settings = manifest["training"]
    environment_seeds = tuple(settings["evaluation_seeds"])
    training_seeds = tuple(settings["training_seeds"])
    rows = []

    def append_episode_rows(policy_name, training_seed, policy):
        for episode_metrics in evaluate_policy(policy, environment_seeds):
            episode_row = {"policy": policy_name, "training_seed": training_seed}
            episode_row.update(episode_metrics)
            rows.append(episode_row)

    for mode in manifest["reward_modes"]:
        for seed in training_seeds:
            with open(f"{results_directory}/q_{mode}_{seed}.json", encoding="utf-8") as file:
                tables = json.load(file)
            agents = tuple(TabularQLearner(seed * 10 + i, TrainingConfig()) for i in range(4))
            for agent, table in zip(agents, tables):
                agent.q = {tuple(map(int, key.split(","))): values for key, values in table.items()}
            append_episode_rows(
                f"iql_{mode}", seed,
                lambda observations, agents=agents: tuple(
                    agent.select_action(observation, greedy=True)
                    for agent, observation in zip(agents, observations)
                ),
            )

    for seed in training_seeds:
        rng = random.Random(seed)
        append_episode_rows("random", seed,
                            lambda observations, rng=rng: tuple(random_action(o, rng)
                                                                for o in observations))
    append_episode_rows("fixed_medium", 0,
                        lambda observations: tuple(fixed_medium_action(o) for o in observations))

    append_episode_rows("threshold", 0,
                        lambda observations: tuple(threshold_action(o) for o in observations))
    run_rows = summarize_by_training_seed(rows)
    policy_groups = defaultdict(list)

    # Each run already averages its held-out episodes; spread is across policy seeds.
    for row in run_rows:
        policy_groups[row["policy"]].append(row)

    summary_rows = []
    for policy, group in sorted(policy_groups.items()):
        summary = {"policy": policy, "runs": len(group)}
        for metric in METRICS:
            values = [row[metric] for row in group]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary_rows.append(summary)

    write_csv(f"{results_directory}/episodes.csv", rows)
    write_csv(f"{results_directory}/run_summary.csv", run_rows)
    write_csv(f"{results_directory}/policy_summary.csv", summary_rows)

    print(f"Saved {len(rows)} held-out episodes and {len(run_rows)} run summaries to {results_directory}")


if __name__ == "__main__":
    main()
