"""Record action choices and a matched held-out episode for the saved baseline."""

import json
import random
from collections import defaultdict

from baselines import fixed_medium_action, random_action, threshold_action
from config import TrainingConfig
from environment import PollutionEnvironment
from evaluate import write_csv
from run_followup import greedy_actions, load_agents


OUTPUT_DIRECTORY = "results/followup/diagnostics"
BASELINE_DIRECTORY = "results/raw"
EXAMPLE_SEED = 100000


def record_policy(policy_name, training_seed, policy, evaluation_seeds, counts, trajectory):
    environment = PollutionEnvironment()
    for demand_seed in evaluation_seeds:
        observations = environment.reset(demand_seed)
        for step in range(environment.config.episode_steps):
            demands = environment.demand
            actions = policy(observations)
            next_observations, _, _, info = environment.step(actions)

            for region in range(4):
                demand, pollution_bin = observations[region]
                key = (policy_name, region, demand, pollution_bin, actions[region])
                counts[key] += 1

                if demand_seed == EXAMPLE_SEED and training_seed in (0, 11):
                    trajectory.append({
                        "policy": policy_name, "training_seed": training_seed,
                        "environment_seed": demand_seed, "step": step + 1, "region": region,
                        "demand": demands[region], "pollution_bin": pollution_bin,
                        "action": actions[region], "emissions": info["emissions"][region],
                        "pollution_load": info["pollution_loads"][region],
                        "mitigation_cost": info["mitigation_costs"][region],
                    })
            observations = next_observations


def main():
    with open(f"{BASELINE_DIRECTORY}/manifest.json", encoding="utf-8") as file:
        manifest = json.load(file)

    settings = TrainingConfig()
    evaluation_seeds = tuple(manifest["training"]["evaluation_seeds"])
    counts = defaultdict(int)
    trajectory = []

    for mode in manifest["reward_modes"]:
        for seed in manifest["training"]["training_seeds"]:
            agents = load_agents(f"{BASELINE_DIRECTORY}/q_{mode}_{seed}.json", seed, settings)
            record_policy(f"iql_{mode}", seed,
                          lambda observations, agents=agents: greedy_actions(observations, agents),
                          evaluation_seeds, counts, trajectory)

    for seed in manifest["training"]["training_seeds"]:
        action_rng = random.Random(seed)
        record_policy("random", seed,
                      lambda observations, rng=action_rng: tuple(random_action(o, rng)
                                                                  for o in observations),
                      evaluation_seeds, counts, trajectory)

    record_policy("fixed_medium", 0,
                  lambda observations: tuple(fixed_medium_action(o) for o in observations),
                  evaluation_seeds, counts, trajectory)
    record_policy("threshold", 0,
                  lambda observations: tuple(threshold_action(o) for o in observations),
                  evaluation_seeds, counts, trajectory)

    count_rows = [
        {"policy": policy, "region": region, "demand": demand,
         "pollution_bin": pollution_bin, "action": action, "count": count}
        for (policy, region, demand, pollution_bin, action), count in sorted(counts.items())
    ]
    write_csv(f"{OUTPUT_DIRECTORY}/action_counts_by_state.csv", count_rows)
    write_csv(f"{OUTPUT_DIRECTORY}/matched_trajectory.csv", trajectory)
    print(f"Saved {len(count_rows)} action counts and {len(trajectory)} example region-steps")


if __name__ == "__main__":
    main()
