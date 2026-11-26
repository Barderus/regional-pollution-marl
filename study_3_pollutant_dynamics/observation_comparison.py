"""Matched mixed-scenario Q-learning comparisons for compact observations."""

import copy
import csv
import os
import random

from study_3_pollutant_dynamics.agents import TabularQLearner
from study_3_pollutant_dynamics.mitigation_parameters import MITIGATION_PROJECTS
from study_3_pollutant_dynamics.rl_experiments import _make_environment, summarize_episode_metrics
from study_3_pollutant_dynamics.rl_formulation import (STATE_BIN_EDGES, build_no_action_reference,
                                                        decode_actions, encode_observation, make_reward_function)
from study_3_pollutant_dynamics.scenario_parameters import get_parameters
from study_3_pollutant_dynamics.train import save_q_table


OBSERVATION_VARIANTS = ("current", "project_status", "project_timing_income")


def project_observation_features(environment, region_index, year, variant):
    projects = environment.projects[region_index]
    active = []
    pending = []

    for project in projects:
        parameters = MITIGATION_PROJECTS[project["category"]][project["pollutant"]]
        activation_year = project["investment_year"] + parameters["delay_years"]
        expiry_year = activation_year + parameters["persistence_years"]

        if activation_year <= year < expiry_year:
            active.append((activation_year, expiry_year))
        elif year < activation_year:
            pending.append((activation_year, expiry_year))

    if variant == "current":
        return {}

    if variant == "project_status":
        active_flag = bool(active)
        pending_flag = bool(pending)
        return {"project_status": int(active_flag) + 2 * int(pending_flag)}

    if variant == "project_timing_income":
        if active and pending:
            timing = 5
        elif pending:
            years_to_activation = min(item[0] - year for item in pending)
            timing = 1 if years_to_activation <= 1 else 2 if years_to_activation <= 3 else 3
        elif active:
            years_to_expiry = min(item[1] - year for item in active)
            timing = 4 if years_to_expiry <= 1 else 5
        else:
            timing = 0
        return {"project_timing": timing}

    raise ValueError("Unknown observation variant: " + str(variant))


def income_change_bin(current_income, previous_income):
    if previous_income is None or previous_income == 0:
        return 1
    change = current_income / previous_income - 1
    return 0 if change < -0.005 else 2 if change > 0.005 else 1


def make_observation(observation, environment, region_index, variant, income_trend=1):
    enriched = dict(observation)
    enriched.update(project_observation_features(environment, region_index, observation["year"], variant))

    if variant == "project_timing_income":
        enriched["income_change"] = income_trend

    return enriched


def observation_bin_edges(variant):
    edges = dict(STATE_BIN_EDGES)

    if variant == "project_status":
        edges["project_status"] = (0, 1, 2)
    elif variant == "project_timing_income":
        edges["project_timing"] = (0, 1, 2, 3, 4)
        edges["income_change"] = (0, 1)
    elif variant != "current":
        raise ValueError("Unknown observation variant: " + str(variant))

    return edges


def _encoded_states(observations, environment, variant, income_trends):
    bin_edges = observation_bin_edges(variant)
    states = []

    for region_index, observation in enumerate(observations):
        enriched = make_observation(observation, environment, region_index, variant, income_trends[region_index])
        states.append(encode_observation(enriched, bin_edges))

    return states


def train_mixed_observation_variant(variant, reward_mode, training_seed, checkpoints, episode_years=40,
                                    epsilon_start=1.0, epsilon_decay=0.9990019209880034,
                                    epsilon_minimum=0.05, learning_rate=0.2, discount_factor=0.95):
    scenarios = ("low", "reference", "high")
    environments = {}
    references = {}

    for scenario in scenarios:
        config, environment = _make_environment(scenario, episode_years, 1.0, 0.6, "constant")
        environments[scenario] = environment
        references[scenario] = build_no_action_reference(config, income_retention_fraction=0.6)

    reward_functions = {scenario: make_reward_function(reward_mode, references[scenario]) for scenario in scenarios}
    seed_generator = random.Random(training_seed)
    agents = [TabularQLearner(8, training_seed * 10 + region, learning_rate, discount_factor, epsilon_start)
              for region in range(4)]
    checkpoints = set(checkpoints)
    checkpoint_agents = {}
    episode_rows = []
    scenario_counts = {scenario: 0 for scenario in scenarios}

    for episode in range(1, max(checkpoints) + 1):
        scenario = seed_generator.choice(scenarios)
        scenario_counts[scenario] += 1
        environment = environments[scenario]
        observations = environment.reset(seed_generator.randrange(1_000_000))
        income_trends = [1] * len(agents)
        previous_gross_income = [None] * len(agents)
        cumulative_rewards = [0.0] * len(agents)
        visited_state_count = 0

        while True:
            states = _encoded_states(observations, environment, variant, income_trends)
            visited_state_count += len(states)
            action_indices = [agent.select_action(state) for agent, state in zip(agents, states)]
            actions = decode_actions(action_indices)
            next_observations, terminated, truncated, info = environment.step(actions)
            rewards = reward_functions[scenario](observations, actions, info, next_observations)
            next_trends = []

            for region, agent in enumerate(agents):
                gross_income = info["regional_economy"][region]["gross_income"]
                next_trend = income_change_bin(gross_income, previous_gross_income[region])
                next_trends.append(next_trend)

            next_states = _encoded_states(next_observations, environment, variant, next_trends)

            for region, agent in enumerate(agents):
                gross_income = info["regional_economy"][region]["gross_income"]
                next_state = next_states[region]
                agent.update(states[region], action_indices[region], rewards[region], next_state,
                             terminated or truncated)
                cumulative_rewards[region] += rewards[region]
                previous_gross_income[region] = gross_income

            income_trends = next_trends
            observations = next_observations
            if terminated or truncated:
                break

        for agent in agents:
            agent.decay_epsilon(epsilon_decay, epsilon_minimum)
        episode_rows.append({"episode": episode, "scenario": scenario, "epsilon": agents[0].epsilon,
                             "visited_states_across_regions": sum(len(agent.q_values) for agent in agents),
                             "state_observations": visited_state_count,
                             "cumulative_reward_all_regions": sum(cumulative_rewards)})

        if episode in checkpoints:
            checkpoint_agents[episode] = copy.deepcopy(agents)

    return checkpoint_agents, episode_rows


def evaluate_observation_agents(agents, variant, scenario, reward_mode, training_seed, training_episodes,
                                episode_years=40):
    config, environment = _make_environment(scenario, episode_years, 1.0, 0.6, "constant")
    reference = build_no_action_reference(config, income_retention_fraction=0.6)
    reward_function = make_reward_function(reward_mode, reference)
    observations = environment.reset(0)
    income_trends = [1] * len(agents)
    previous_gross_income = [None] * len(agents)
    unseen_visits = [0] * len(agents)
    unseen_states = [set() for agent in agents]
    action_counts = [{action: 0 for action in environment.action_names} for agent in agents]
    trajectory = []
    total_rewards = [0.0] * len(agents)

    while True:
        states = _encoded_states(observations, environment, variant, income_trends)
        action_indices = []

        for region, (agent, state) in enumerate(zip(agents, states)):
            if state not in agent.q_values:
                unseen_visits[region] += 1
                unseen_states[region].add(state)
            action_indices.append(agent.select_action(state, greedy=True))

        actions = decode_actions(action_indices)
        next_observations, terminated, truncated, info = environment.step(actions)
        rewards = reward_function(observations, actions, info, next_observations)
        next_trends = []

        for region, action in enumerate(actions):
            action_counts[region][action] += 1
            total_rewards[region] += rewards[region]
            gross_income = info["regional_economy"][region]["gross_income"]
            next_trends.append(income_change_bin(gross_income, previous_gross_income[region]))
            previous_gross_income[region] = gross_income

        trajectory.append(info)
        observations = next_observations
        income_trends = next_trends
        if terminated or truncated:
            break

    summary = summarize_episode_metrics(trajectory, environment)
    summary.update({"observation_variant": variant, "evaluation_scenario": scenario,
                    "reward_mode": reward_mode, "training_seed": training_seed,
                    "training_episodes": training_episodes, "unseen_state_visits": sum(unseen_visits),
                    "unseen_state_visit_fraction": sum(unseen_visits) / (len(agents) * episode_years),
                    "unique_unseen_states": sum(len(states) for states in unseen_states),
                    "unique_training_states": sum(len(agent.q_values) for agent in agents),
                    "cumulative_reward_all_regions": sum(total_rewards),
                    "action_counts_by_region": action_counts})

    return summary


def run_observation_comparison(results_directory, training_seeds=(11, 23, 37), checkpoints=(300, 1000, 3000),
                               reward_modes=("local", "global", "mixed"), evaluation_scenarios=("low", "reference", "high")):
    os.makedirs(results_directory + "/q_tables", exist_ok=True)
    training_rows = []
    evaluation_rows = []
    action_rows = []

    for variant in OBSERVATION_VARIANTS:
        for reward_mode in reward_modes:
            for training_seed in training_seeds:
                checkpoint_agents, episode_rows = train_mixed_observation_variant(
                    variant, reward_mode, training_seed, checkpoints
                )

                for row in episode_rows:
                    row.update({"observation_variant": variant, "reward_mode": reward_mode,
                                "training_seed": training_seed})
                training_rows.extend(episode_rows)

                for checkpoint, agents in checkpoint_agents.items():
                    for region, agent in enumerate(agents):
                        filename = (results_directory + "/q_tables/q_" + variant + "_" + reward_mode + "_"
                                    + str(training_seed) + "_ep_" + str(checkpoint) + "_region_" + str(region) + ".json")
                        save_q_table(agent, filename)

                    for scenario in evaluation_scenarios:
                        summary = evaluate_observation_agents(agents, variant, scenario, reward_mode,
                                                              training_seed, checkpoint)
                        counts = summary.pop("action_counts_by_region")
                        evaluation_rows.append(summary)

                        for region, region_counts in enumerate(counts):
                            for action, count in region_counts.items():
                                action_rows.append({"observation_variant": variant, "reward_mode": reward_mode,
                                                    "training_seed": training_seed, "training_episodes": checkpoint,
                                                    "evaluation_scenario": scenario, "region_index": region,
                                                    "action": action, "count": count, "fraction": count / 40})

    for filename, rows in (("training_history.csv", training_rows), ("evaluation.csv", evaluation_rows),
                           ("action_frequencies.csv", action_rows)):
        if rows:
            with open(results_directory + "/" + filename, "w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    return training_rows, evaluation_rows, action_rows
