"""Independent PPO training and evaluation helpers for Study 3."""

import random

from study_3_pollutant_dynamics.dqn_agent import encode_dqn_observation
from study_3_pollutant_dynamics.ppo_agent import PPOAgent
from study_3_pollutant_dynamics.rl_experiments import summarize_episode_metrics
from study_3_pollutant_dynamics.rl_formulation import STATE_BIN_EDGES


def train_ppo_agents(environments, reward_functions, training_seed, training_episodes,
                     learning_rate=0.0003, discount_factor=0.99, gae_lambda=0.95, clip_ratio=0.2,
                     entropy_coefficient=0.01, value_coefficient=0.5, update_epochs=4,
                     minibatch_size=64, hidden_size=128, progress_callback=None):
    scenarios = tuple(environments)
    if not scenarios:
        raise ValueError("At least one training scenario is required")
    if set(scenarios) != set(reward_functions):
        raise ValueError("Each training scenario needs its own reward function")

    first_environment = environments[scenarios[0]]
    agents = [PPOAgent(len(STATE_BIN_EDGES), len(first_environment.action_names), training_seed * 10 + region,
                       learning_rate, discount_factor, gae_lambda, clip_ratio, entropy_coefficient,
                       value_coefficient, update_epochs, minibatch_size, hidden_size)
              for region in range(len(first_environment.regions))]
    scenario_generator = random.Random(training_seed)
    episode_rows = []
    scenario_counts = {scenario: 0 for scenario in scenarios}

    for episode in range(1, training_episodes + 1):
        scenario = scenario_generator.choice(scenarios)
        scenario_counts[scenario] += 1
        episode_seed = scenario_generator.randrange(1_000_000)
        environment = environments[scenario]
        observations = environment.reset(episode_seed)
        episode_rewards = [0.0] * len(agents)
        finished = False

        while not finished:
            decisions = [agent.act(observation) for agent, observation in zip(agents, observations)]
            actions = [environment.action_names[decision[1]] for decision in decisions]
            next_observations, terminated, truncated, info = environment.step(actions)
            rewards = reward_functions[scenario](observations, actions, info, next_observations)
            finished = terminated or truncated
            for region, (agent, decision) in enumerate(zip(agents, decisions)):
                state, action, old_log_probability, value = decision
                agent.remember(state, action, rewards[region], old_log_probability, value, finished)
                episode_rewards[region] += rewards[region]
            observations = next_observations

        for agent in agents:
            agent.update()

        row = {"episode": episode, "seed": episode_seed, "pollutant_scenario": scenario,
               "mean_cumulative_reward_per_region": sum(episode_rewards) / len(episode_rewards),
               "cumulative_reward_all_regions": sum(episode_rewards), "epsilon": 0.0,
               "visited_states_across_regions": sum(len(agent.training_states) for agent in agents),
               "scenario_counts": dict(scenario_counts)}
        episode_rows.append(row)
        if progress_callback is not None:
            progress_callback(row, training_episodes)

    return agents, episode_rows


def evaluate_ppo_agents(environment, agents, reward_function, seed=0):
    observations = environment.reset(seed)
    episode_rewards = [0.0] * len(agents)
    action_counts = [{action: 0 for action in environment.action_names} for _ in agents]
    evaluation_states = [set() for _ in agents]
    trajectory = []
    finished = False

    while not finished:
        states = [encode_dqn_observation(observation) for observation in observations]
        actions_indices = []
        for region, (agent, state) in enumerate(zip(agents, states)):
            evaluation_states[region].add(state)
            actions_indices.append(agent.select_action(observations[region], deterministic=True))
        actions = [environment.action_names[action] for action in actions_indices]
        next_observations, terminated, truncated, info = environment.step(actions)
        rewards = reward_function(observations, actions, info, next_observations)
        for region, action in enumerate(actions):
            action_counts[region][action] += 1
            episode_rewards[region] += rewards[region]
        trajectory.append(info)
        observations = next_observations
        finished = terminated or truncated

    summary = summarize_episode_metrics(trajectory, environment)
    summary["cumulative_reward_all_regions"] = sum(episode_rewards)
    summary["mean_reward_per_region"] = sum(episode_rewards) / len(episode_rewards)
    summary["unique_training_states"] = sum(agent.training_state_count for agent in agents)
    summary["unique_evaluation_states"] = sum(len(states) for states in evaluation_states)
    summary["action_counts_by_region"] = action_counts
    summary["trajectory"] = trajectory
    return summary
