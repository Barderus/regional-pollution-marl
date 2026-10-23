import json
import random

from study_2_economic_sustainability.agents import TabularQLearner
from study_2_economic_sustainability.baselines import choose_baseline_actions
from study_2_economic_sustainability.config import TrainingConfig
from study_2_economic_sustainability.environment import EconomicSustainabilityEnvironment


def run_episode(environment, seed, policy, agents=None, policy_rng=None,
                train=False, keep_trajectory=False):
    """Run one episode and return its common metrics and optional trajectory.

    Give either four agents or the name of a baseline policy. Training updates Q-values and evaluation uses greedy actions and leaves Q-values unchanged.
    """
    if agents is not None and len(agents) != 4:
        raise ValueError("Exactly four agents are required")
    if train and agents is None:
        raise ValueError("Training requires four agents")

    observations = environment.reset(seed)
    starting_balances = environment.currency_balances
    action_counts = [0, 0, 0, 0, 0]
    total_income = 0.0
    mitigation_spending = 0.0
    total_emissions = 0.0
    invalid_attempts = 0
    gpi_sum = 0.0
    maximum_pollution = max(environment.pollution_loads)
    trajectory = []

    while True:
        if agents is None:
            actions = choose_baseline_actions(policy, environment, observations, policy_rng)
        else:
            actions = tuple(
                agents[region].select_action(observations[region], greedy=not train)
                for region in range(4)
            )

        next_observations, rewards, terminated, truncated, info = environment.step(actions)

        if train:
            for region in range(4):
                agents[region].update(
                    observations[region], actions[region], rewards[region],
                    next_observations[region], terminated,
                )

        for region in range(4):
            action_counts[actions[region]] += 1
            total_income += info["income"][region]
            total_emissions += info["base_emissions"][region]
            invalid_attempts += info["invalid_attempts"][region]
            if actions[region] in (2, 3, 4):
                mitigation_spending += info["costs_paid"][region]

        gpi_sum += info["global_pollution_index"]
        maximum_pollution = max(maximum_pollution, max(info["pollution_loads"]))

        if keep_trajectory:
            point = {
                "step": environment.step_count,
                "gpi": info["global_pollution_index"],
                "maximum_regional_pollution": max(info["pollution_loads"]),
                "mean_edi": sum(info["development_indices"]) / 4,
                "total_currency": sum(info["currency_balances"]),
                "base_emissions": sum(info["base_emissions"]),
                "mitigation_spending": sum(
                    info["costs_paid"][region]
                    for region in range(4)
                    if actions[region] in (2, 3, 4)
                ),
            }
            for region in range(4):
                point[f"pollution_{region}"] = info["pollution_loads"][region]
                point[f"edi_{region}"] = info["development_indices"][region]
                point[f"currency_{region}"] = info["currency_balances"][region]
            trajectory.append(point)

        observations = next_observations
        if terminated or truncated:
            break

    result = {
        "policy": policy,
        "environment_seed": seed,
        "steps": environment.step_count,
        "terminated": terminated,
        "truncated": truncated,
        "mean_gpi": gpi_sum / environment.step_count,
        "maximum_pollution": maximum_pollution,
        "mean_final_edi": sum(environment.development_indices) / 4,
        "total_income": total_income,
        "mitigation_spending": mitigation_spending,
        "total_base_emissions": total_emissions,
        "invalid_attempts": invalid_attempts,
    }

    for region in range(4):
        result[f"initial_currency_{region}"] = starting_balances[region]
        result[f"final_currency_{region}"] = environment.currency_balances[region]
        result[f"final_edi_{region}"] = environment.development_indices[region]
        result[f"final_pollution_{region}"] = environment.pollution_loads[region]

    for action in range(5):
        result[f"action_{action}_count"] = action_counts[action]

    return result, trajectory


def train_agents(reward_mode, training_seed, episodes, config=None, environment_config=None):
    settings = config or TrainingConfig()
    environment = EconomicSustainabilityEnvironment(reward_mode, environment_config)
    episode_rng = random.Random(training_seed)
    agents = tuple(TabularQLearner(training_seed * 10 + region, settings) for region in range(4))
    training_rows = []

    # These training episode seeds stay below the held-out evaluation range.
    for episode in range(episodes):
        environment_seed = episode_rng.randrange(100_000)
        result, _ = run_episode(environment, environment_seed, reward_mode, agents, train=True)
        result["training_seed"] = training_seed
        result["episode"] = episode + 1
        training_rows.append(result)

        for agent in agents:
            agent.decay_epsilon()

    return agents, training_rows


def evaluate_agents(agents, reward_mode, environment_seeds, environment_config=None):
    environment = EconomicSustainabilityEnvironment(reward_mode, environment_config)
    rows = []

    for seed in environment_seeds:
        result, _ = run_episode(environment, seed, reward_mode, agents)
        rows.append(result)

    return rows


def save_agents(filename, agents):
    tables = []

    for agent in agents:
        table = {}
        for observation, values in sorted(agent.q.items()):
            key = ",".join(str(bin_value) for bin_value in observation)
            table[key] = values
        tables.append(table)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(tables, file, indent=2)
        file.write("\n")


def load_agents(filename, config=None):
    with open(filename, "r", encoding="utf-8") as file:
        tables = json.load(file)

    if len(tables) != 4:
        raise ValueError("A saved policy must contain four Q-tables")

    agents = []
    for region, table in enumerate(tables):
        agent = TabularQLearner(region, config)
        for key, values in table.items():
            observation = tuple(int(bin_value) for bin_value in key.split(","))
            agent.q[observation] = values
        agents.append(agent)

    return tuple(agents)
