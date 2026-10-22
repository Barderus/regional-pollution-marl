"""Run the prespecified seed expansion and spillover comparison."""

import json
import random
import statistics
from collections import defaultdict

from study_1_emissions_coordination.agents import TabularQLearner
from study_1_emissions_coordination.baselines import fixed_medium_action, random_action, threshold_action
from study_1_emissions_coordination.config import EnvironmentConfig, TrainingConfig
from study_1_emissions_coordination.evaluate import evaluate_policy, summarize_by_policy, summarize_by_training_seed, write_csv
from study_1_emissions_coordination.train import REWARD_MODES, save_agents, train_iql


TRAINING_SEEDS = (11, 23, 37, 41, 53, 67, 71, 83, 97, 101, 107,
                  109, 127, 131, 137, 149, 157, 163, 173, 179)
SCENARIOS = (("spillover_02", 0.2), ("spillover_00", 0.0))
OUTPUT_DIRECTORY = "study_1_emissions_coordination/results/followup"


def load_agents(filename, training_seed, settings):
    with open(filename, encoding="utf-8") as file:
        tables = json.load(file)

    agents = tuple(TabularQLearner(training_seed * 10 + i, settings) for i in range(4))
    for agent, table in zip(agents, tables):
        agent.q = {tuple(map(int, key.split(","))): values for key, values in table.items()}
    return agents


def greedy_actions(observations, agents):
    return tuple(agent.select_action(observation, greedy=True)
                 for agent, observation in zip(agents, observations))


def paired_differences(run_rows, scenario):
    by_run = {(row["policy"], row["training_seed"]): row for row in run_rows}
    rows = []
    for seed in TRAINING_SEEDS:
        global_run = by_run[("iql_global", seed)]
        local_run = by_run[("iql_local", seed)]
        for metric in ("mean_regional_pollution", "mitigation_cost", "threshold_exceedance"):
            rows.append({"scenario": scenario, "training_seed": seed, "metric": metric,
                         "global_minus_local": global_run[metric] - local_run[metric]})
    return rows


def run_scenario(name, spillover, settings):
    output_directory = f"{OUTPUT_DIRECTORY}/{name}"
    environment_config = EnvironmentConfig()
    environment_config.spillover = spillover

    manifest = {
        "environment": vars(environment_config),
        "training": vars(settings),
        "command": "python -m study_1_emissions_coordination.run_followup",
        "reward_modes": list(REWARD_MODES),
        "models": [f"q_{mode}_{seed}.json" for mode in REWARD_MODES
                   for seed in settings.training_seeds],
        "evaluation": "20 common held-out demand seeds per policy and training seed",
    }

    for mode in REWARD_MODES:
        for seed in settings.training_seeds:
            agents = train_iql(mode, seed, settings, environment_config)
            save_agents(f"{output_directory}/q_{mode}_{seed}.json", agents)
            print(f"{name}: trained {mode}, seed {seed}", flush=True)

    with open(f"{output_directory}/manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")

    episode_rows = []

    def add_episodes(policy_name, training_seed, policy):
        for metrics in evaluate_policy(policy, settings.evaluation_seeds, environment_config):
            row = {"policy": policy_name, "training_seed": training_seed}
            row.update(metrics)
            episode_rows.append(row)

    for mode in REWARD_MODES:
        for seed in settings.training_seeds:
            agents = load_agents(f"{output_directory}/q_{mode}_{seed}.json", seed, settings)
            add_episodes(f"iql_{mode}", seed,
                         lambda observations, agents=agents: greedy_actions(observations, agents))

    for seed in settings.training_seeds:
        action_rng = random.Random(seed)
        add_episodes("random", seed,
                     lambda observations, rng=action_rng: tuple(random_action(o, rng)
                                                                 for o in observations))

    add_episodes("fixed_medium", 0,
                 lambda observations: tuple(fixed_medium_action(o) for o in observations))
    add_episodes("threshold", 0,
                 lambda observations: tuple(threshold_action(o) for o in observations))

    run_rows = summarize_by_training_seed(episode_rows)
    summary_rows = summarize_by_policy(run_rows)
    write_csv(f"{output_directory}/episodes.csv", episode_rows)
    write_csv(f"{output_directory}/run_summary.csv", run_rows)
    write_csv(f"{output_directory}/policy_summary.csv", summary_rows)
    print(f"{name}: saved {len(episode_rows)} episodes and {len(run_rows)} runs", flush=True)
    return paired_differences(run_rows, name), summary_rows


def summarize_ablation(difference_rows):
    values = defaultdict(dict)
    for row in difference_rows:
        values[(row["training_seed"], row["metric"])][row["scenario"]] = row["global_minus_local"]

    changes = defaultdict(list)
    for (seed, metric), scenarios in values.items():
        changes[metric].append(scenarios["spillover_02"] - scenarios["spillover_00"])

    summaries = []
    for metric in ("mean_regional_pollution", "mitigation_cost", "threshold_exceedance"):
        for scenario in ("spillover_02", "spillover_00"):
            items = [row["global_minus_local"] for row in difference_rows
                     if row["metric"] == metric and row["scenario"] == scenario]
            summaries.append({"metric": metric, "comparison": scenario,
                              "paired_runs": len(items), "mean": statistics.mean(items),
                              "sd": statistics.stdev(items)})
        items = changes[metric]
        summaries.append({"metric": metric, "comparison": "spillover_02_minus_00",
                          "paired_runs": len(items), "mean": statistics.mean(items),
                          "sd": statistics.stdev(items)})
    return summaries


def main():
    settings = TrainingConfig()
    settings.training_seeds = TRAINING_SEEDS

    difference_rows = []
    for name, spillover in SCENARIOS:
        paired_rows, _ = run_scenario(name, spillover, settings)
        difference_rows.extend(paired_rows)

    write_csv(f"{OUTPUT_DIRECTORY}/paired_differences.csv", difference_rows)
    write_csv(f"{OUTPUT_DIRECTORY}/ablation_summary.csv", summarize_ablation(difference_rows))
    print("Saved paired spillover comparison", flush=True)


if __name__ == "__main__":
    main()
