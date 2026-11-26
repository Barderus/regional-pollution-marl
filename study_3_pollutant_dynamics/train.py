"""Independent tabular Q-Learning training and evaluation helpers for Study 3."""

import json
import os
import random

from study_3_pollutant_dynamics.agents import TabularQLearner
from study_3_pollutant_dynamics.policy_evaluation import summarize_integrated_episode


def run_agent_episode(environment, agents, seed, state_encoder, action_decoder, reward_function, update_q_values=False, greedy=True):
    observations = environment.reset(seed)
    total_rewards = [0.0] * len(agents)
    action_counts = []
    trajectory = []

    for agent in agents:
        region_counts = {}
        for action_name in environment.action_names:
            region_counts[action_name] = 0
        action_counts.append(region_counts)

    while True:
        states = []
        for observation in observations:
            states.append(state_encoder(observation))

        action_indices = []
        for region, agent in enumerate(agents):
            action_indices.append(agent.select_action(states[region], greedy=greedy))

        regional_actions = action_decoder(action_indices)
        next_observations, terminated, truncated, info = environment.step(regional_actions)
        rewards = reward_function(observations, regional_actions, info, next_observations)
        episode_finished = terminated or truncated

        for region, agent in enumerate(agents):
            next_state = state_encoder(next_observations[region])
            if update_q_values:
                agent.update(states[region], action_indices[region], rewards[region], next_state, episode_finished)
            total_rewards[region] += rewards[region]
            action_counts[region][regional_actions[region]] += 1

        trajectory.append(info)
        observations = next_observations

        if episode_finished:
            break

    return total_rewards, trajectory, action_counts


def train_episode(environment, agents, seed, state_encoder, action_decoder, reward_function):
    return run_agent_episode(environment, agents, seed, state_encoder, action_decoder, reward_function, True, False)


def train_agents(environment, action_count, training_seed, training_episodes, state_encoder, action_decoder, reward_function,
                 learning_rate=0.2, discount_factor=0.95, epsilon_start=1.0, epsilon_decay=0.99, epsilon_minimum=0.05):
    seed_generator = random.Random(training_seed)
    agents = []

    for region in range(len(environment.config.regions)):
        agent_seed = training_seed * 10 + region
        agents.append(TabularQLearner(action_count, agent_seed, learning_rate, discount_factor, epsilon_start))

    episode_rows = []

    for episode in range(training_episodes):
        episode_seed = seed_generator.randrange(1_000_000)
        episode_epsilon = agents[0].epsilon
        rewards, trajectory, action_counts = train_episode(environment, agents, episode_seed, state_encoder, action_decoder,
                                                           reward_function)
        summary = summarize_integrated_episode("independent_q_learning", trajectory, environment)
        summary["episode"] = episode + 1
        summary["seed"] = episode_seed
        summary["mean_cumulative_reward_per_region"] = sum(rewards) / len(rewards)
        summary["cumulative_reward_all_regions"] = sum(rewards)
        summary["epsilon"] = episode_epsilon
        summary["action_counts_by_region"] = action_counts
        episode_rows.append(summary)

        for agent in agents:
            agent.decay_epsilon(epsilon_decay, epsilon_minimum)

    return agents, episode_rows


def save_q_table(agent, filename):
    rows = []

    for state, values in agent.q_values.items():
        visits = []
        for action in range(agent.action_count):
            visits.append(agent.action_visits.get((state, action), 0))
        rows.append({"state": list(state), "q_values": values, "action_visits": visits})

    with open(filename, "w", encoding="utf-8") as output_file:
        json.dump(rows, output_file)


def load_q_table(agent, filename):
    with open(filename, "r", encoding="utf-8") as input_file:
        rows = json.load(input_file)

    for row in rows:
        state = tuple(row["state"])
        agent.q_values[state] = row["q_values"]
        for action, visits in enumerate(row["action_visits"]):
            if visits:
                agent.action_visits[(state, action)] = visits


def save_policy_q_tables(agents, reward_mode, training_seed, results_directory):
    os.makedirs(results_directory, exist_ok=True)

    for region, agent in enumerate(agents):
        filename = results_directory + "/q_" + reward_mode + "_" + str(training_seed) + "_region_" + str(region) + ".json"
        save_q_table(agent, filename)
