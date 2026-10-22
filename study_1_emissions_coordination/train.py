# Train four independent Q-learners per reward condition and save their tables

import json
import random

from study_1_emissions_coordination.agents import TabularQLearner
from study_1_emissions_coordination.config import EnvironmentConfig, TrainingConfig
from study_1_emissions_coordination.environment import PollutionEnvironment

REWARD_MODES = ("local", "global", "mixed")


def train_iql(reward_mode, training_seed, config=None, environment_config=None,
              externality_weight=0.0):
    settings = config or TrainingConfig()

    if reward_mode not in REWARD_MODES + ("externality",):
        raise ValueError(reward_mode)

    environment = PollutionEnvironment(reward_mode, environment_config, externality_weight)
    episode_rng = random.Random(training_seed)
    agents = tuple(TabularQLearner(training_seed * 10 + i, settings) for i in range(4))

    # Evaluation seeds start at 100,000, outside this training seed range.
    for _ in range(settings.training_episodes):
        observation = environment.reset(episode_rng.randrange(100_000))
        while True:
            actions = tuple(agent.select_action(obs) for agent, obs in zip(agents, observation))
            next_observation, rewards, done, _ = environment.step(actions)
            for i, agent in enumerate(agents):
                agent.update(observation[i], actions[i], rewards[i], next_observation[i], done)
            observation = next_observation
            if done:
                break
        for agent in agents:
            agent.decay_epsilon()
    return agents


def save_agents(filename, agents):
    tables = []
    for agent in agents:
        table = {}
        for (demand, pollution_bin), values in sorted(agent.q.items()):
            table[f"{demand},{pollution_bin}"] = values
        tables.append(table)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(tables, file, indent=2)
        file.write("\n")


def main():
    output_directory = "study_1_emissions_coordination/results/raw"
    config = TrainingConfig()

    manifest = {
        "environment": vars(EnvironmentConfig()),
        "training": vars(config),
        "command": "python -m study_1_emissions_coordination.train",
        "reward_modes": list(REWARD_MODES),
        "models": [f"q_{mode}_{seed}.json" for mode in REWARD_MODES for seed in config.training_seeds],
        "hypotheses": ["Shared rewards may reduce total exposure.",
                       "Local rewards may have lower mitigation cost."],
    }

    for mode in REWARD_MODES:
        for seed in config.training_seeds:
            agents = train_iql(mode, seed, config)
            filename = f"{output_directory}/q_{mode}_{seed}.json"
            save_agents(filename, agents)
            print(f"Trained {mode}, seed {seed}", flush=True)

    with open(f"{output_directory}/manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
