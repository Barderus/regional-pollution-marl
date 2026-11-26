"""Training checkpoints and diagnostics for Study 3 tabular Q-learning experiments."""

import copy
import os
import random

from study_3_pollutant_dynamics.agents import TabularQLearner
from study_3_pollutant_dynamics.config import EnvironmentConfig
from study_3_pollutant_dynamics.integrated_environment import Study3IntegratedEnvironment
from study_3_pollutant_dynamics.policy_evaluation import (FIXED_POLICY_NAMES, choose_fixed_policy_actions,
                                                          run_fixed_policy_episode)
from study_3_pollutant_dynamics.rl_formulation import (STATE_BIN_EDGES, build_no_action_reference, decode_actions,
                                                       encode_observation, make_reward_function)
from study_3_pollutant_dynamics.scenario_parameters import get_parameters
from study_3_pollutant_dynamics.train import save_q_table, train_episode


def make_state_encoder(bin_edges):
    def state_encoder(observation):
        return encode_observation(observation, bin_edges)

    return state_encoder


def train_agents_with_checkpoints(environment, action_count, training_seed, checkpoints, state_encoder,
                                  action_decoder, reward_function, learning_rate=0.2, discount_factor=0.95,
                                  epsilon_start=1.0, epsilon_decay=0.99, epsilon_minimum=0.05):
    checkpoint_set = set(checkpoints)
    if not checkpoint_set or min(checkpoint_set) <= 0:
        raise ValueError("Training checkpoints must be positive episode counts")
    seed_generator = random.Random(training_seed)
    agents = []
    for region in range(len(environment.config.regions)):
        agents.append(TabularQLearner(action_count, training_seed * 10 + region, learning_rate, discount_factor,
                                      epsilon_start))
    checkpoint_agents = {}
    episode_rows = []

    for episode in range(1, max(checkpoint_set) + 1):
        episode_seed = seed_generator.randrange(1_000_000)
        epsilon = agents[0].epsilon
        rewards, _, _ = train_episode(environment, agents, episode_seed, state_encoder, action_decoder,
                                      reward_function)
        episode_rows.append({"episode": episode, "seed": episode_seed,
                             "mean_cumulative_reward_per_region": sum(rewards) / len(rewards),
                             "cumulative_reward_all_regions": sum(rewards), "epsilon": epsilon,
                             "visited_states_across_regions": sum(len(agent.q_values) for agent in agents),
                             "visited_state_action_pairs": sum(len(agent.action_visits) for agent in agents)})
        for agent in agents:
            agent.decay_epsilon(epsilon_decay, epsilon_minimum)
        if episode in checkpoint_set:
            checkpoint_agents[episode] = copy.deepcopy(agents)

    return checkpoint_agents, episode_rows


def summarize_episode_metrics(trajectory, environment):
    from study_3_pollutant_dynamics.policy_evaluation import summarize_integrated_episode

    summary = summarize_integrated_episode("policy", trajectory, environment)
    gross_income = 0.0
    net_income = 0.0
    edi_sum = 0.0
    pollution_values = {"PM2.5": [], "CO2": [], "Nitrate-N": [], "Lead": []}
    for info in trajectory:
        gross_income += sum(result["gross_income"] for result in info["regional_economy"])
        net_income += sum(result["net_income"] for result in info["regional_economy"])
        edi_sum += sum(result["EDI"] for result in info["regional_economy"])
        pollution_values["PM2.5"].extend(info["stocks"]["PM2.5"])
        pollution_values["CO2"].append(info["stocks"]["CO2"])
        pollution_values["Nitrate-N"].extend(1000 * stock / environment.config.nitrate["water_volume_m3"]
                                               for stock in info["stocks"]["Nitrate-N"])
        pollution_values["Lead"].extend(info["stocks"]["Lead"])
    region_years = len(trajectory) * len(environment.regions)
    summary["cumulative_gross_income"] = gross_income
    summary["cumulative_net_income"] = net_income
    summary["average_EDI"] = edi_sum / region_years
    summary["mean_CO2_ppm"] = sum(pollution_values["CO2"]) / len(pollution_values["CO2"])
    summary["mean_PM2.5_ug_m3"] = sum(pollution_values["PM2.5"]) / len(pollution_values["PM2.5"])
    summary["mean_nitrate_mg_N_L"] = sum(pollution_values["Nitrate-N"]) / len(pollution_values["Nitrate-N"])
    summary["mean_lead_total_kg"] = sum(pollution_values["Lead"]) / len(pollution_values["Lead"])
    return summary


def _make_environment(pollution_scenario, episode_years, starting_currency_multiplier, income_retention_fraction,
                      development_cost_shape):
    parameters = get_parameters(scenarios={"pm25": pollution_scenario, "co2": pollution_scenario,
                                          "nitrate": pollution_scenario, "lead": pollution_scenario},
                                episode_steps=episode_years)
    config = EnvironmentConfig(parameters)
    environment = Study3IntegratedEnvironment(config, starting_currency_multiplier=starting_currency_multiplier,
                                              development_cost_shape=development_cost_shape,
                                              income_retention_fraction=income_retention_fraction)
    return config, environment


def run_training_coverage_diagnostics(results_directory, training_scenarios, evaluation_scenarios, reward_modes,
                                       training_seeds, checkpoints, bin_edges, episode_years=40,
                                       income_retention_fraction=0.6, starting_currency_multiplier=1.0,
                                       development_cost_shape="constant", epsilon_start=1.0, epsilon_decay=0.99,
                                       epsilon_minimum=0.05, learning_rate=0.2, discount_factor=0.95,
                                       include_fixed_baselines=True, encoding_name="current", unseen_action_rules=(None,)):
    os.makedirs(results_directory + "/q_tables", exist_ok=True)
    encoder = make_state_encoder(bin_edges)
    references = {}
    configs = {}
    for scenario in evaluation_scenarios:
        config, environment = _make_environment(scenario, episode_years, starting_currency_multiplier,
                                                income_retention_fraction, development_cost_shape)
        configs[scenario] = config
        references[scenario] = build_no_action_reference(config, starting_currency_multiplier,
                                                          income_retention_fraction)

    training_rows = []
    evaluation_rows = []
    trajectory_rows = []
    action_rows = []
    fixed_rows = []
    if include_fixed_baselines:
        for scenario in evaluation_scenarios:
            config = configs[scenario]
            for policy_name in FIXED_POLICY_NAMES:
                _, environment = _make_environment(scenario, episode_years, starting_currency_multiplier,
                                                    income_retention_fraction, development_cost_shape)
                trajectory, terminated, truncated = run_fixed_policy_episode(policy_name, environment, 0)
                summary = summarize_episode_metrics(trajectory, environment)
                summary.update({"policy_type": "fixed", "policy": policy_name, "evaluation_scenario": scenario,
                                "completed_horizon": truncated, "terminated": terminated,
                                "encoding": "not_applicable"})
                fixed_rows.append(summary)
                counts = [{action: 0 for action in environment.action_names} for region in environment.regions]
                for info in trajectory:
                    for region, result in enumerate(info["regional_economy"]):
                        counts[region][result["action"]] += 1
                for region, region_counts in enumerate(counts):
                    for action, count in region_counts.items():
                        action_rows.append({"policy_type": "fixed", "policy": policy_name,
                                            "evaluation_scenario": scenario, "region": environment.regions[region],
                                            "action": action, "count": count, "fraction": count / episode_years,
                                            "encoding": "not_applicable"})
                for step, info in enumerate(trajectory, start=1):
                    for region, result in enumerate(info["regional_economy"]):
                        row = _trajectory_row("fixed", policy_name, scenario, scenario, None, None, None,
                                               encoding_name, step, region, result, info, config)
                        row["unseen_action_rule"] = "not_applicable"
                        trajectory_rows.append(row)

    for training_scenario in training_scenarios:
        reward_functions = {}
        for reward_mode in reward_modes:
            reward_functions[reward_mode] = make_reward_function(reward_mode, references[training_scenario])
        for reward_mode in reward_modes:
            for training_seed in training_seeds:
                _, environment = _make_environment(training_scenario, episode_years, starting_currency_multiplier,
                                                    income_retention_fraction, development_cost_shape)
                checkpoint_agents, episode_rows = train_agents_with_checkpoints(
                    environment, len(environment.action_names), training_seed, checkpoints, encoder, decode_actions,
                    reward_functions[reward_mode], learning_rate, discount_factor, epsilon_start, epsilon_decay,
                    epsilon_minimum,
                )
                for row in episode_rows:
                    row.update({"training_scenario": training_scenario, "reward_mode": reward_mode,
                                "training_seed": training_seed, "encoding": encoding_name})
                training_rows.extend(episode_rows)
                for checkpoint in checkpoints:
                    agents = checkpoint_agents[checkpoint]
                    for region, agent in enumerate(agents):
                        filename = (results_directory + "/q_tables/q_" + encoding_name + "_" + training_scenario + "_"
                                    + reward_mode + "_" + str(training_seed) + "_ep_" + str(checkpoint) + "_region_"
                                    + str(region) + ".json")
                        save_q_table(agent, filename)
                    for evaluation_scenario in evaluation_scenarios:
                        _, evaluation_environment = _make_environment(
                            evaluation_scenario, episode_years, starting_currency_multiplier,
                            income_retention_fraction, development_cost_shape,
                        )
                        reward_function = make_reward_function(reward_mode, references[evaluation_scenario])
                        for unseen_action in unseen_action_rules:
                            summary = evaluate_agent_policy(evaluation_environment, agents, encoder, decode_actions,
                                                            reward_function, unseen_action=unseen_action)
                            trajectory = summary.pop("trajectory")
                            counts = summary.pop("action_counts_by_region")
                            summary.update({"policy_type": "learned", "policy": "q_" + reward_mode + "_" + str(training_seed),
                                            "training_scenario": training_scenario, "evaluation_scenario": evaluation_scenario,
                                            "training_seed": training_seed, "training_episodes": checkpoint,
                                            "reward_mode": reward_mode, "encoding": encoding_name})
                            evaluation_rows.append(summary)
                            for region, region_counts in enumerate(counts):
                                for action, count in region_counts.items():
                                    action_rows.append({"policy_type": "learned", "policy": summary["policy"],
                                                        "training_scenario": training_scenario,
                                                        "evaluation_scenario": evaluation_scenario,
                                                        "training_seed": training_seed, "training_episodes": checkpoint,
                                                        "reward_mode": reward_mode, "encoding": encoding_name,
                                                        "unseen_action_rule": summary["unseen_action_rule"],
                                                        "region": evaluation_environment.regions[region], "action": action,
                                                        "count": count, "fraction": count / episode_years})
                            for step, info in enumerate(trajectory, start=1):
                                for region, result in enumerate(info["regional_economy"]):
                                    row = _trajectory_row(
                                        "learned", summary["policy"], training_scenario, evaluation_scenario,
                                        training_seed, checkpoint, reward_mode, encoding_name, step, region, result,
                                        info, evaluation_environment.config,
                                    )
                                    row["unseen_action_rule"] = summary["unseen_action_rule"]
                                    trajectory_rows.append(row)

    return training_rows, evaluation_rows, trajectory_rows, action_rows, fixed_rows


def _trajectory_row(policy_type, policy, training_scenario, evaluation_scenario, training_seed, training_episodes,
                    reward_mode, encoding, step, region, region_result, info, config):
    nitrate_mg_l = 1000 * info["stocks"]["Nitrate-N"][region] / config.nitrate["water_volume_m3"]
    return {"policy_type": policy_type, "policy": policy, "training_scenario": training_scenario,
            "evaluation_scenario": evaluation_scenario, "training_seed": training_seed,
            "training_episodes": training_episodes, "reward_mode": reward_mode, "encoding": encoding,
            "step": step, "region": region_result["region"], "action": region_result["action"],
            "EDI": region_result["EDI"], "currency": region_result["currency"],
            "gross_income": region_result["gross_income"], "net_income": region_result["net_income"],
            "PM2.5_ug_m3": info["stocks"]["PM2.5"][region], "CO2_ppm": info["stocks"]["CO2"],
            "nitrate_mg_N_L": nitrate_mg_l, "lead_total_kg": info["stocks"]["Lead"][region],
            "development_spending": region_result["development_cost"],
            "mitigation_spending": region_result["investment_cost"]}


def evaluate_agent_policy(environment, agents, state_encoder, action_decoder, reward_function, seed=0, unseen_action=None):
    observations = environment.reset(seed)
    rewards = [0.0] * len(agents)
    actions_by_region = []
    unseen_visits_by_region = [0] * len(agents)
    total_state_visits = [0] * len(agents)
    unseen_states_by_region = [set() for agent in agents]
    trajectories = []
    for agent in agents:
        actions_by_region.append({action: 0 for action in environment.action_names})

    while True:
        states = [state_encoder(observation) for observation in observations]
        action_indices = []
        for region, agent in enumerate(agents):
            state = states[region]
            total_state_visits[region] += 1
            if state not in agent.q_values:
                unseen_visits_by_region[region] += 1
                unseen_states_by_region[region].add(state)
            if state not in agent.q_values and unseen_action is not None:
                if unseen_action not in environment.action_names:
                    raise ValueError("Unknown unseen-state action: " + str(unseen_action))
                action_indices.append(environment.action_names.index(unseen_action))
            else:
                action_indices.append(agent.select_action(state, greedy=True))
        regional_actions = action_decoder(action_indices)
        next_observations, terminated, truncated, info = environment.step(regional_actions)
        step_rewards = reward_function(observations, regional_actions, info, next_observations)
        for region, action in enumerate(regional_actions):
            rewards[region] += step_rewards[region]
            actions_by_region[region][action] += 1
        trajectories.append(info)
        observations = next_observations
        if terminated or truncated:
            break

    summary = summarize_episode_metrics(trajectories, environment)
    summary["cumulative_reward_all_regions"] = sum(rewards)
    summary["mean_reward_per_region"] = sum(rewards) / len(rewards)
    summary["evaluation_state_visits"] = sum(total_state_visits)
    summary["unseen_state_visits"] = sum(unseen_visits_by_region)
    summary["unseen_state_visit_fraction"] = summary["unseen_state_visits"] / summary["evaluation_state_visits"]
    summary["unique_unseen_states"] = sum(len(states) for states in unseen_states_by_region)
    summary["unique_training_states"] = sum(len(agent.q_values) for agent in agents)
    summary["unseen_action_rule"] = unseen_action or "default_greedy_tie"
    summary["action_counts_by_region"] = actions_by_region
    summary["trajectory"] = trajectories
    return summary


def train_agents_across_scenarios(environments, action_count, training_seed, checkpoints, state_encoder,
                                  action_decoder, reward_functions, learning_rate=0.2, discount_factor=0.95,
                                  epsilon_start=1.0, epsilon_decay=0.999, epsilon_minimum=0.05):
    checkpoint_set = set(checkpoints)
    if not checkpoint_set or min(checkpoint_set) <= 0:
        raise ValueError("Training checkpoints must be positive episode counts")
    scenarios = tuple(environments)
    if not scenarios or set(scenarios) != set(reward_functions):
        raise ValueError("Each training scenario must have an environment and reward function")
    seed_generator = random.Random(training_seed)
    agents = []
    region_count = len(next(iter(environments.values())).config.regions)
    for region in range(region_count):
        agents.append(TabularQLearner(action_count, training_seed * 10 + region, learning_rate, discount_factor,
                                      epsilon_start))
    checkpoint_agents = {}
    episode_rows = []
    scenario_counts = {scenario: 0 for scenario in scenarios}

    for episode in range(1, max(checkpoint_set) + 1):
        scenario = seed_generator.choice(scenarios)
        scenario_counts[scenario] += 1
        episode_seed = seed_generator.randrange(1_000_000)
        epsilon = agents[0].epsilon
        rewards, _, _ = train_episode(environments[scenario], agents, episode_seed, state_encoder, action_decoder,
                                      reward_functions[scenario])
        episode_rows.append({"episode": episode, "seed": episode_seed, "pollutant_scenario": scenario,
                             "mean_cumulative_reward_per_region": sum(rewards) / len(rewards),
                             "cumulative_reward_all_regions": sum(rewards), "epsilon": epsilon,
                             "visited_states_across_regions": sum(len(agent.q_values) for agent in agents),
                             "visited_state_action_pairs": sum(len(agent.action_visits) for agent in agents)})
        for agent in agents:
            agent.decay_epsilon(epsilon_decay, epsilon_minimum)
        if episode in checkpoint_set:
            snapshot = copy.deepcopy(agents)
            for agent in snapshot:
                agent.scenario_counts = dict(scenario_counts)
            checkpoint_agents[episode] = snapshot

    return checkpoint_agents, episode_rows


def run_mixed_scenario_diagnostics(results_directory, evaluation_scenarios, reward_modes, training_seeds,
                                  checkpoints, bin_edges, episode_years=40, income_retention_fraction=0.6,
                                  starting_currency_multiplier=1.0, development_cost_shape="constant",
                                  epsilon_start=1.0, epsilon_decay=0.999, epsilon_minimum=0.05,
                                  learning_rate=0.2, discount_factor=0.95, unseen_action_rules=(None,)):
    os.makedirs(results_directory + "/q_tables", exist_ok=True)
    encoder = make_state_encoder(bin_edges)
    environments = {}
    configs = {}
    references = {}
    for scenario in evaluation_scenarios:
        config, environment = _make_environment(scenario, episode_years, starting_currency_multiplier,
                                                income_retention_fraction, development_cost_shape)
        configs[scenario] = config
        environments[scenario] = environment
        references[scenario] = build_no_action_reference(config, starting_currency_multiplier,
                                                          income_retention_fraction)

    training_rows = []
    evaluation_rows = []
    trajectory_rows = []
    action_rows = []
    for reward_mode in reward_modes:
        reward_functions = {scenario: make_reward_function(reward_mode, references[scenario])
                            for scenario in evaluation_scenarios}
        for training_seed in training_seeds:
            checkpoint_agents, episode_rows = train_agents_across_scenarios(
                environments, len(next(iter(environments.values())).action_names), training_seed, checkpoints,
                encoder, decode_actions, reward_functions, learning_rate, discount_factor, epsilon_start,
                epsilon_decay, epsilon_minimum,
            )
            for row in episode_rows:
                row.update({"training_scenario": "mixed", "reward_mode": reward_mode,
                            "training_seed": training_seed, "encoding": "current"})
            training_rows.extend(episode_rows)
            for checkpoint in checkpoints:
                agents = checkpoint_agents[checkpoint]
                for region, agent in enumerate(agents):
                    filename = (results_directory + "/q_tables/q_mixed_" + reward_mode + "_" + str(training_seed)
                                + "_ep_" + str(checkpoint) + "_region_" + str(region) + ".json")
                    save_q_table(agent, filename)
                for evaluation_scenario in evaluation_scenarios:
                    reward_function = reward_functions[evaluation_scenario]
                    for unseen_action in unseen_action_rules:
                        _, evaluation_environment = _make_environment(
                            evaluation_scenario, episode_years, starting_currency_multiplier,
                            income_retention_fraction, development_cost_shape,
                        )
                        summary = evaluate_agent_policy(evaluation_environment, agents, encoder, decode_actions,
                                                        reward_function, unseen_action=unseen_action)
                        trajectory = summary.pop("trajectory")
                        counts = summary.pop("action_counts_by_region")
                        summary.update({"policy_type": "learned", "policy": "mixed_q_" + reward_mode + "_" + str(training_seed),
                                        "training_scenario": "mixed", "evaluation_scenario": evaluation_scenario,
                                        "training_seed": training_seed, "training_episodes": checkpoint,
                                        "reward_mode": reward_mode, "encoding": "current"})
                        evaluation_rows.append(summary)
                        for region, region_counts in enumerate(counts):
                            for action, count in region_counts.items():
                                action_rows.append({"policy_type": "learned", "policy": summary["policy"],
                                                    "training_scenario": "mixed", "evaluation_scenario": evaluation_scenario,
                                                    "training_seed": training_seed, "training_episodes": checkpoint,
                                                    "reward_mode": reward_mode, "encoding": "current",
                                                    "unseen_action_rule": summary["unseen_action_rule"],
                                                    "region": evaluation_environment.regions[region], "action": action,
                                                    "count": count, "fraction": count / episode_years})
                        for step, info in enumerate(trajectory, start=1):
                            for region, result in enumerate(info["regional_economy"]):
                                row = _trajectory_row(
                                    "learned", summary["policy"], "mixed", evaluation_scenario, training_seed,
                                    checkpoint, reward_mode, "current", step, region, result, info,
                                    evaluation_environment.config,
                                )
                                row["unseen_action_rule"] = summary["unseen_action_rule"]
                                trajectory_rows.append(row)
    return training_rows, evaluation_rows, trajectory_rows, action_rows


def rebuild_evaluation_trajectories(results_directory, scenario_evaluation_rows, mixed_evaluation_rows,
                                    episode_years=40, income_retention_fraction=0.6,
                                    starting_currency_multiplier=1.0, development_cost_shape="constant"):
    from study_3_pollutant_dynamics.train import load_q_table

    trajectory_rows = []
    all_evaluations = list(scenario_evaluation_rows) + list(mixed_evaluation_rows)
    for row in all_evaluations:
        evaluation_scenario = row["evaluation_scenario"]
        config, environment = _make_environment(evaluation_scenario, episode_years, starting_currency_multiplier,
                                                income_retention_fraction, development_cost_shape)
        reference = build_no_action_reference(config, starting_currency_multiplier, income_retention_fraction)
        policy_type = row["training_scenario"]
        q_directory = results_directory + "/mixed_scenarios/q_tables" if policy_type == "mixed" else (
            results_directory + "/scenario_specific/q_tables"
        )
        training_scenario = policy_type if policy_type != "mixed" else "mixed"
        agents = []
        for region in range(len(config.regions)):
            agent = TabularQLearner(8, int(row["training_seed"]) * 10 + region)
            if policy_type == "mixed":
                filename = (q_directory + "/q_mixed_" + row["reward_mode"] + "_" + str(int(row["training_seed"]))
                            + "_ep_" + str(int(row["training_episodes"])) + "_region_" + str(region) + ".json")
            else:
                filename = (q_directory + "/q_current_" + training_scenario + "_" + row["reward_mode"] + "_"
                            + str(int(row["training_seed"])) + "_ep_" + str(int(row["training_episodes"]))
                            + "_region_" + str(region) + ".json")
            load_q_table(agent, filename)
            agents.append(agent)
        unseen_action = None if row["unseen_action_rule"] == "default_greedy_tie" else row["unseen_action_rule"]
        reward_function = make_reward_function(row["reward_mode"], reference)
        summary = evaluate_agent_policy(environment, agents, make_state_encoder(STATE_BIN_EDGES), decode_actions,
                                        reward_function, unseen_action=unseen_action)
        for step, info in enumerate(summary["trajectory"], start=1):
            for region, result in enumerate(info["regional_economy"]):
                trajectory_row = _trajectory_row("learned", row["policy"], training_scenario, evaluation_scenario,
                                                 int(row["training_seed"]), int(row["training_episodes"]),
                                                 row["reward_mode"], "current", step, region, result, info,
                                                 environment.config)
                trajectory_row["unseen_action_rule"] = row["unseen_action_rule"]
                trajectory_rows.append(trajectory_row)
    return trajectory_rows
