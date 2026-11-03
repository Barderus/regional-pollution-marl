import json
import random

from study_2_economic_sustainability.agents import TabularQLearner
from study_2_economic_sustainability.config import EnvironmentConfig, TrainingConfig
from study_2_economic_sustainability.environment import (EconomicSustainabilityEnvironment)

def train_episode(environment, agents, seed, reward_transform=None):
    observations = environment.reset(seed)

    while True:
        actions = []

        for region in environment.config.regions:
            action = agents[region].select_action(observations[region])
            actions.append(action)

        next_observations, rewards, terminated, truncated, info = environment.step(actions)

        if reward_transform is not None:
            rewards = reward_transform(environment, rewards, info, terminated)

        for region in environment.config.regions:
            agents[region].update(
                observations[region],
                actions[region],
                rewards[region],
                next_observations[region],
                terminated,
            )

        observations = next_observations

        if terminated or truncated:
            break


def train_agents(reward_mode, training_seed, environment_config=None, training_config=None, reward_transform=None):
    if environment_config is None:
        environment_config = EnvironmentConfig()

    if training_config is None:
        training_config = TrainingConfig()

    environment = EconomicSustainabilityEnvironment(reward_mode, environment_config)
    episode_seeds = random.Random(training_seed)
    agents = []

    for region in environment_config.regions:
        agent_seed = training_seed * 10 + region
        agents.append(TabularQLearner(agent_seed, training_config))

    for _ in range(training_config.training_episodes):
        episode_seed = episode_seeds.randrange(100_000)
        train_episode(environment, agents, episode_seed, reward_transform)

        for agent in agents:
            agent.decay_epsilon()

    return agents


def save_q_tables(trained_policies, results_directory="results", notebook_number=3):
    saved_policies = {}

    for (reward_mode, training_seed), agents in trained_policies.items():
        regional_tables = []

        for agent in agents:
            table = {}

            for observation, values in sorted(agent.q.items()):
                observation_key = ""

                for value in observation:
                    if observation_key:
                        observation_key += ","
                    observation_key += str(value)

                table[observation_key] = values

            regional_tables.append(table)

        policy_key = f"{reward_mode}|{training_seed}"
        saved_policies[policy_key] = regional_tables

    filename = f"{results_directory}/q_tables_notebook{notebook_number:02}.json"

    with open(filename, "w", encoding="utf-8") as result_file:
        json.dump(saved_policies, result_file, indent=2)
        result_file.write("\n")


def load_q_tables(
    reward_modes,
    training_seeds,
    training_config=None,
    results_directory="results",
    notebook_number=3,
):
    if training_config is None:
        training_config = TrainingConfig()

    filename = f"{results_directory}/q_tables_notebook{notebook_number:02}.json"

    with open(filename, "r", encoding="utf-8") as result_file:
        saved_policies = json.load(result_file)

    trained_policies = {}

    for reward_mode in reward_modes:
        for training_seed in training_seeds:
            policy_key = f"{reward_mode}|{training_seed}"
            regional_tables = saved_policies[policy_key]
            agents = []

            for region, table in enumerate(regional_tables):
                agent_seed = training_seed * 10 + region
                agent = TabularQLearner(agent_seed, training_config)

                for key, values in table.items():
                    observation = []

                    for value in key.split(","):
                        observation.append(int(value))

                    agent.q[tuple(observation)] = values

                agents.append(agent)

            trained_policies[(reward_mode, training_seed)] = agents

    return trained_policies
