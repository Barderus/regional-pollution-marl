"""Focused checks for the Study 3 tabular RL formulation."""

from study_3_pollutant_dynamics.config import EnvironmentConfig
from study_3_pollutant_dynamics.integrated_environment import Study3IntegratedEnvironment
from study_3_pollutant_dynamics.rl_formulation import (ACTION_NAMES, STATE_BIN_EDGES, build_no_action_reference,
                                                        calculate_reward_components, decode_actions, encode_observation,
                                                        rewards_for_mode)
from study_3_pollutant_dynamics.rl_experiments import (evaluate_agent_policy, make_state_encoder,
                                                       train_agents_across_scenarios,
                                                       train_agents_with_checkpoints)
from study_3_pollutant_dynamics.scenario_parameters import get_parameters
import os

from study_3_pollutant_dynamics.agents import TabularQLearner
from study_3_pollutant_dynamics.train import load_q_table, save_q_table, train_agents


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def test_encoded_observation_has_eight_discrete_features():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    observations = environment.reset(0)
    state = encode_observation(observations[0])
    feature_count = len(STATE_BIN_EDGES)
    _require(len(state) == feature_count == 8, "The encoded state must contain eight features")
    integer_bins = True
    for value in state:
        if type(value) is not int:
            integer_bins = False
    _require(integer_bins, "Every encoded feature must be an integer bin")


def test_action_decoder_covers_all_eight_actions():
    decoded_actions = decode_actions(list(range(len(ACTION_NAMES))))
    _require(tuple(decoded_actions) == ACTION_NAMES, "Action indices must map to the environment action names")

    invalid_action_rejected = False
    try:
        decode_actions([len(ACTION_NAMES)])
    except ValueError:
        invalid_action_rejected = True

    _require(invalid_action_rejected, "An out-of-range action must be rejected")


def test_no_action_reference_has_zero_pollution_pressure():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    reference_states = build_no_action_reference(config, income_retention_fraction=0.6)
    environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    observations = environment.reset(0)
    next_observations, terminated, truncated, info = environment.step(["maintain"] * 4)
    components = calculate_reward_components(observations, next_observations, info, reference_states)

    for regional_component in components["regional_components"]:
        _require(abs(regional_component["pollution_pressure"]) < 1e-12, "The reference path must have zero pollution pressure")
        _require(abs(regional_component["local_reward"]) < 1e-12, "No-action development and pollution rewards must be zero")


def test_reward_modes_match_their_equations():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    reference_states = build_no_action_reference(config, income_retention_fraction=0.6)
    environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    observations = environment.reset(0)
    actions = ["develop"] * 4
    next_observations, terminated, truncated, info = environment.step(actions)
    components = calculate_reward_components(observations, next_observations, info, reference_states)
    local_rewards = rewards_for_mode("local", observations, next_observations, info, reference_states)
    global_rewards = rewards_for_mode("global", observations, next_observations, info, reference_states)
    mixed_rewards = rewards_for_mode("mixed", observations, next_observations, info, reference_states)
    expected_global = sum(local_rewards) / len(local_rewards)

    _require(local_rewards == components["local_rewards"], "Local rewards must match the regional components")
    shared_global_reward = True
    for value in global_rewards:
        if abs(value - expected_global) >= 1e-12:
            shared_global_reward = False
    _require(shared_global_reward, "Global rewards must be shared")

    for region, mixed_reward in enumerate(mixed_rewards):
        expected_mixed = 0.7 * local_rewards[region] + 0.3 * expected_global
        _require(abs(mixed_reward - expected_mixed) < 1e-12, "Mixed rewards must combine local and global values")


def test_time_is_included_in_the_environment_observation():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    observations = environment.reset(0)
    _require(observations[0]["year"] == 0, "The reset observation must identify year zero")
    observations, terminated, truncated, info = environment.step(["maintain"] * 4)
    _require(observations[0]["year"] == 1, "The next observation must identify the completed year")


def test_unaffordable_development_does_not_spend_currency_or_raise_edi():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    environment = Study3IntegratedEnvironment(config, starting_currency_multiplier=0.0, income_retention_fraction=0.6)
    observations = environment.reset(0)
    initial_edi = []
    for observation in observations:
        initial_edi.append(observation["EDI"])
    observations, terminated, truncated, info = environment.step(["develop"] * 4)

    for region, result in enumerate(info["regional_economy"]):
        _require(not result["development_affordable"], "Development must be blocked without opening currency")
        _require(result["development_cost"] == 0, "Unaffordable development must not be charged")
        _require(observations[region]["EDI"] == initial_edi[region], "Unaffordable development must leave EDI unchanged")
        _require(observations[region]["currency"] >= 0, "Currency must not become negative")


def test_training_is_reproducible_for_a_fixed_seed():
    config = EnvironmentConfig(get_parameters(episode_steps=3))
    reference_states = build_no_action_reference(config, income_retention_fraction=0.6)

    def reward_function(observations, actions, info, next_observations):
        return rewards_for_mode("mixed", observations, next_observations, info, reference_states)

    environment_a = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    environment_b = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    agents_a, rows_a = train_agents(environment_a, len(ACTION_NAMES), 21, 4, encode_observation, decode_actions, reward_function)
    agents_b, rows_b = train_agents(environment_b, len(ACTION_NAMES), 21, 4, encode_observation, decode_actions, reward_function)

    for region in range(len(agents_a)):
        _require(agents_a[region].q_values == agents_b[region].q_values, "Q tables must match for the same training seed")
    _require(rows_a == rows_b, "Training episode logs must match for the same training seed")


def test_greedy_selection_does_not_add_unseen_states():
    agent = TabularQLearner(3, 7)
    action = agent.select_action((9, 9), greedy=True)
    _require(action == 0, "An unseen greedy state must use the deterministic first-action tie-break")
    _require((9, 9) not in agent.q_values, "Greedy evaluation must not change the learned Q-table")


def test_checkpoint_training_and_evaluation_track_unseen_states():
    config = EnvironmentConfig(get_parameters(episode_steps=2))
    reference_states = build_no_action_reference(config, income_retention_fraction=0.6)
    encoder = make_state_encoder(STATE_BIN_EDGES)

    def reward_function(observations, actions, info, next_observations):
        return rewards_for_mode("local", observations, next_observations, info, reference_states)

    training_environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    checkpoints, episode_rows = train_agents_with_checkpoints(training_environment, len(ACTION_NAMES), 31, [1, 2],
                                                               encoder, decode_actions, reward_function)
    _require(set(checkpoints) == {1, 2}, "Training must save each requested checkpoint")
    _require(len(episode_rows) == 2, "Training must log the requested episode count")

    evaluation_environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    evaluation_agents = [TabularQLearner(len(ACTION_NAMES), region) for region in range(4)]
    evaluation = evaluate_agent_policy(evaluation_environment, evaluation_agents, encoder, decode_actions,
                                       reward_function)
    _require(evaluation["evaluation_state_visits"] == 8, "Coverage must count each region-year observation")
    _require(evaluation["unseen_state_visits"] == 8, "Untrained policies must report all evaluation states unseen")
    _require(all(not agent.q_values for agent in evaluation_agents), "Evaluation must not add states to Q-tables")


def test_mixed_scenario_training_samples_each_scenario_reproducibly():
    scenarios = ("low", "reference", "high")
    environments = {}
    reward_functions = {}
    for scenario in scenarios:
        config = EnvironmentConfig(get_parameters(scenarios={"pm25": scenario, "co2": scenario,
                                                              "nitrate": scenario, "lead": scenario},
                                                  episode_steps=2))
        environments[scenario] = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
        reference_states = build_no_action_reference(config, income_retention_fraction=0.6)
        reward_functions[scenario] = lambda observations, actions, info, next_observations, ref=reference_states: (
            rewards_for_mode("local", observations, next_observations, info, ref)
        )
    checkpoints_a, rows_a = train_agents_across_scenarios(environments, len(ACTION_NAMES), 51, [9],
                                                           encode_observation, decode_actions, reward_functions,
                                                           epsilon_decay=0.9)
    environments_b = {}
    for scenario in scenarios:
        config = EnvironmentConfig(get_parameters(scenarios={"pm25": scenario, "co2": scenario,
                                                              "nitrate": scenario, "lead": scenario},
                                                  episode_steps=2))
        environments_b[scenario] = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    checkpoints_b, rows_b = train_agents_across_scenarios(environments_b, len(ACTION_NAMES), 51, [9],
                                                           encode_observation, decode_actions, reward_functions,
                                                           epsilon_decay=0.9)
    _require(set(row["pollutant_scenario"] for row in rows_a) == set(scenarios),
             "Mixed training must sample every configured pollutant scenario")
    _require(rows_a == rows_b, "Mixed-scenario episode sampling must reproduce for a fixed seed")
    for region in range(4):
        _require(checkpoints_a[9][region].q_values == checkpoints_b[9][region].q_values,
                 "Mixed-scenario Q-tables must reproduce for a fixed seed")


def test_alternative_unseen_action_rule_changes_only_unseen_decisions():
    config = EnvironmentConfig(get_parameters(episode_steps=2))
    reference_states = build_no_action_reference(config, income_retention_fraction=0.6)
    reward_function = lambda observations, actions, info, next_observations: rewards_for_mode(
        "local", observations, next_observations, info, reference_states
    )
    agents = [TabularQLearner(len(ACTION_NAMES), region) for region in range(4)]
    maintain_environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    balanced_environment = Study3IntegratedEnvironment(config, income_retention_fraction=0.6)
    maintain_result = evaluate_agent_policy(maintain_environment, agents, encode_observation, decode_actions,
                                            reward_function)
    balanced_result = evaluate_agent_policy(balanced_environment, agents, encode_observation, decode_actions,
                                            reward_function, unseen_action="develop_balanced")
    _require(maintain_result["action_counts_by_region"][0]["maintain"] == 2,
             "The default tie-break must maintain at unseen states")
    _require(balanced_result["action_counts_by_region"][0]["develop_balanced"] == 2,
             "The alternate rule must use the selected fallback at unseen states")
    _require(balanced_result["final_mean_EDI"] > maintain_result["final_mean_EDI"],
             "The alternate unseen-state action should change episode outcomes")
    _require(all(not agent.q_values for agent in agents), "Neither evaluation rule may modify Q-tables")


def test_q_table_round_trip_preserves_values_and_visits():
    filename = "study_3_pollutant_dynamics/tests/q_table_roundtrip_tmp.json"
    agent = TabularQLearner(3, 7)
    agent.q_values[(1, 2, 3)] = [0.25, -0.5, 1.75]
    agent.action_visits[((1, 2, 3), 2)] = 9
    save_q_table(agent, filename)
    restored_agent = TabularQLearner(3, 8)
    load_q_table(restored_agent, filename)
    os.remove(filename)
    _require(restored_agent.q_values == agent.q_values, "Saved Q-values must be restored")
    _require(restored_agent.action_visits == agent.action_visits, "Saved action visits must be restored")


if __name__ == "__main__":
    test_encoded_observation_has_eight_discrete_features()
    test_action_decoder_covers_all_eight_actions()
    test_no_action_reference_has_zero_pollution_pressure()
    test_reward_modes_match_their_equations()
    test_time_is_included_in_the_environment_observation()
    test_unaffordable_development_does_not_spend_currency_or_raise_edi()
    test_training_is_reproducible_for_a_fixed_seed()
    test_greedy_selection_does_not_add_unseen_states()
    test_checkpoint_training_and_evaluation_track_unseen_states()
    test_mixed_scenario_training_samples_each_scenario_reproducibly()
    test_alternative_unseen_action_rule_changes_only_unseen_decisions()
    test_q_table_round_trip_preserves_values_and_visits()
    print("Study 3 RL formulation checks passed.")
