"""DQN training and evaluation helpers for the annual Study 3 environment."""

import random
from study_3_pollutant_dynamics.dqn_agent import DQNAgent, encode_dqn_observation
from study_3_pollutant_dynamics.rl_experiments import summarize_episode_metrics
from study_3_pollutant_dynamics.rl_formulation import STATE_BIN_EDGES


def train_dqn_agents(environments, reward_functions, training_seed, training_episodes,
                     learning_rate=0.001, discount_factor=0.95, epsilon_start=1.0,
                     epsilon_decay=0.999, epsilon_minimum=0.05, replay_buffer_size=20000,
                     batch_size=64, learning_starts=256, train_frequency=4,
                     target_update_frequency=250, hidden_size=128, progress_callback=None):
    scenarios = tuple(environments)

    if not scenarios:
        raise ValueError("At least one training scenario is required")

    if set(scenarios) != set(reward_functions):
        raise ValueError("Each training scenario needs its own reward function")

    first_environment = environments[scenarios[0]]
    agents = []

    for region in range(len(first_environment.regions)):
        agent_seed = training_seed * 10 + region
        agent = DQNAgent(len(STATE_BIN_EDGES), len(first_environment.action_names), agent_seed, learning_rate, discount_factor,
                         epsilon_start, replay_buffer_size, batch_size, learning_starts, train_frequency,
                         target_update_frequency, hidden_size)
        agents.append(agent)

    scenario_generator = random.Random(training_seed)
    episode_rows = []
    scenario_counts = {}

    for scenario in scenarios:
        scenario_counts[scenario] = 0

    for episode in range(1, training_episodes + 1):
        scenario = scenario_generator.choice(scenarios)
        scenario_counts[scenario] += 1
        episode_seed = scenario_generator.randrange(1_000_000)
        environment = environments[scenario]
        observations = environment.reset(episode_seed)
        episode_rewards = [0.0] * len(agents)
        finished = False

        while not finished:
            states = []

            for region, observation in enumerate(observations):
                state = encode_dqn_observation(observation)
                agents[region].training_states.add(state)
                states.append(state)

            action_indices = []

            for region, agent in enumerate(agents):
                action_indices.append(agent.select_action(states[region]))

            actions = []

            for action_index in action_indices:
                actions.append(environment.action_names[action_index])

            next_observations, terminated, truncated, info = environment.step(actions)
            rewards = reward_functions[scenario](observations, actions, info, next_observations)
            next_states = []

            for observation in next_observations:
                next_states.append(encode_dqn_observation(observation))

            finished = terminated or truncated

            for region, agent in enumerate(agents):
                agent.update(states[region], action_indices[region], rewards[region], next_states[region], finished)
                episode_rewards[region] += rewards[region]

            observations = next_observations

        for agent in agents:
            agent.decay_epsilon(epsilon_decay, epsilon_minimum)

        mean_reward = sum(episode_rewards) / len(episode_rewards)
        visited_states = sum(len(agent.training_states) for agent in agents)
        mean_epsilon = sum(agent.epsilon for agent in agents) / len(agents)
        row = {"episode": episode, "seed": episode_seed, "pollutant_scenario": scenario,
               "mean_cumulative_reward_per_region": mean_reward,
               "cumulative_reward_all_regions": sum(episode_rewards), "epsilon": mean_epsilon,
               "visited_states_across_regions": visited_states, "scenario_counts": dict(scenario_counts)}
        episode_rows.append(row)

        if progress_callback is not None:
            progress_callback(row, training_episodes)

    return agents, episode_rows


def evaluate_dqn_agents(environment, agents, reward_function, seed=0):
    observations = environment.reset(seed)
    episode_rewards = [0.0] * len(agents)
    action_counts = []
    trajectory = []

    for agent in agents:
        counts = {}

        for action in environment.action_names:
            counts[action] = 0

        action_counts.append(counts)

    evaluation_states = [set() for agent in agents]
    finished = False

    while not finished:
        states = []

        for observation in observations:
            states.append(encode_dqn_observation(observation))

        action_indices = []

        for region, agent in enumerate(agents):
            evaluation_states[region].add(states[region])
            action_indices.append(agent.select_action(states[region], greedy=True))

        actions = []

        for action_index in action_indices:
            actions.append(environment.action_names[action_index])

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
    training_state_counts = []

    for agent in agents:
        training_state_count = getattr(agent, "training_state_count", len(agent.training_states))
        training_state_counts.append(training_state_count)

    summary["unique_training_states"] = sum(training_state_counts)
    summary["unique_evaluation_states"] = sum(len(states) for states in evaluation_states)
    summary["action_counts_by_region"] = action_counts
    summary["trajectory"] = trajectory
    return summary
