"""Interactive terminal entry point for Study 3 RL training and evaluation.

Run from the repository root with:
python study_3_pollutant_dynamics/run_training.py
"""

import json
import os
from datetime import datetime

from study_3_pollutant_dynamics.agents import TabularQLearner
from study_3_pollutant_dynamics.policy_evaluation import run_fixed_policy_episode
from study_3_pollutant_dynamics.rl_experiments import (_make_environment, evaluate_agent_policy,
                                                       make_state_encoder, summarize_episode_metrics,
                                                       train_agents_across_scenarios, train_agents_with_checkpoints)
from study_3_pollutant_dynamics.rl_formulation import (ACTION_NAMES, STATE_BIN_EDGES, build_no_action_reference,
                                                       decode_actions, make_reward_function)
from study_3_pollutant_dynamics.train import load_q_table, save_q_table


RUNS_DIRECTORY = "../results/cli_runs"
REWARD_MODES = ("local", "global", "mixed")
POLLUTION_SCENARIOS = ("low", "reference", "high")
EPISODE_YEARS = 40
INCOME_RETENTION = 0.6
DEVELOPMENT_COST_SHAPE = "constant"
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
EPSILON_START = 1.0
EPSILON_MINIMUM = 0.05
DQN_SETTINGS = {"learning_rate": 0.001, "replay_buffer_size": 20000, "batch_size": 64, "learning_starts": 256, "train_frequency": 4,
                "target_update_frequency": 250, "hidden_size": 128}
PPO_SETTINGS = {"learning_rate": 0.0003, "gae_lambda": 0.95, "clip_ratio": 0.2, "entropy_coefficient": 0.01,
                "value_coefficient": 0.5, "update_epochs": 4, "minibatch_size": 64, "hidden_size": 128}


def choose_option(prompt, choices, default=None):
    print(prompt)

    for number, label in choices.items():
        print(str(number) + ") " + label)

    while True:
        answer = input("Choose an option: ")

        if answer == "" and default is not None:
            return default

        if answer in choices:
            return answer

        print("Enter one of the listed option numbers.")


def choose_integer(prompt, default, minimum=1):
    while True:
        answer = input(prompt + " [" + str(default) + "] ")

        if answer == "":
            return default

        try:
            value = int(answer)
        except ValueError:
            print("Enter a whole number.")
            continue

        if value >= minimum:
            return value

        print("Enter a value of at least " + str(minimum) + ".")


def choose_training_settings():
    algorithm_choice = choose_option("Available algorithms:", {"1": "Independent tabular Q-Learning",
                                                                "2": "Independent Deep Q-Network (DQN)",
                                                                "3": "Independent Proximal Policy Optimization (PPO)"}, "1")
    reward_choice = choose_option("Reward structure:", {"1": "Local", "2": "Global", "3": "Mixed"}, "3")
    scenario_choice = choose_option("Training conditions:", {"1": "Mixed low, reference, and high pollution",
                                                               "2": "Reference pollution only",
                                                               "3": "Low pollution only",
                                                               "4": "High pollution only"}, "1")
    reward_mode = REWARD_MODES[int(reward_choice) - 1]
    algorithm_name = "independent_tabular_q_learning"
    learning_rate = LEARNING_RATE

    if algorithm_choice == "2":
        algorithm_name = "independent_dqn"
        learning_rate = DQN_SETTINGS["learning_rate"]
    elif algorithm_choice == "3":
        algorithm_name = "independent_ppo"
        learning_rate = PPO_SETTINGS["learning_rate"]

    training_scenarios = ["low", "reference", "high"]

    if scenario_choice != "1":
        scenario_by_choice = {"2": "reference", "3": "low", "4": "high"}
        training_scenarios = [scenario_by_choice[scenario_choice]]

    settings = {"algorithm": algorithm_name, "reward_mode": reward_mode,
                "training_scenarios": training_scenarios,
                "training_episodes": choose_integer("Training episodes", 1000),
                "training_seed": choose_integer("Training seed", 11, 0),
                "episode_years": EPISODE_YEARS, "income_retention_fraction": INCOME_RETENTION,
                "development_cost_shape": DEVELOPMENT_COST_SHAPE, "learning_rate": learning_rate,
                "discount_factor": DISCOUNT_FACTOR, "epsilon_start": EPSILON_START,
                "epsilon_minimum": EPSILON_MINIMUM}

    if algorithm_name == "independent_dqn":
        settings.update(DQN_SETTINGS)
    elif algorithm_name == "independent_ppo":
        settings.update(PPO_SETTINGS)

    settings["epsilon_decay"] = (EPSILON_MINIMUM / EPSILON_START) ** (1 / settings["training_episodes"])
    settings["state_bin_edges"] = STATE_BIN_EDGES
    settings["action_names"] = ACTION_NAMES
    settings["algorithm_menu_choice"] = algorithm_choice
    return settings


def show_training_progress(row, total_episodes, default_scenario="mixed sampled"):
    episode = row["episode"]
    interval = max(1, total_episodes // 100)

    if episode == 1 or episode % interval == 0 or episode == total_episodes:
        reward = row.get("mean_cumulative_reward_per_region", 0.0)
        epsilon = row["epsilon"]
        scenario = row.get("pollutant_scenario", row.get("scenario", default_scenario))
        progress = episode / total_episodes * 100
        message = "\rTraining {0:.0f}% | episode {1}/{2} | scenario {3} | reward {4:.2f} | epsilon {5:.3f}"
        print(message.format(progress, episode, total_episodes, scenario, reward, epsilon), end="", flush=True)


def train_selected_model(settings):
    if settings["algorithm"] == "independent_dqn":
        return train_dqn_model(settings)
    if settings["algorithm"] == "independent_ppo":
        return train_ppo_model(settings)

    episodes = settings["training_episodes"]
    training_scenarios = settings["training_scenarios"]
    encoder = make_state_encoder(STATE_BIN_EDGES)
    checkpoint = [episodes]
    agents = None
    history = []

    if len(training_scenarios) == 1:
        scenario = training_scenarios[0]
        config, environment = _make_environment(scenario, EPISODE_YEARS, 1.0, INCOME_RETENTION,
                                                DEVELOPMENT_COST_SHAPE)
        reference_states = build_no_action_reference(config, income_retention_fraction=INCOME_RETENTION)
        reward_function = make_reward_function(settings["reward_mode"], reference_states)
        checkpoints, history = train_agents_with_checkpoints(
            environment, len(ACTION_NAMES), settings["training_seed"], checkpoint, encoder, decode_actions,
            reward_function, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, settings["epsilon_decay"],
            EPSILON_MINIMUM, lambda row, total: show_training_progress(row, total, scenario),
        )
        agents = checkpoints[episodes]
    else:
        environments = {}
        reward_functions = {}

        for scenario in training_scenarios:
            config, environment = _make_environment(scenario, EPISODE_YEARS, 1.0, INCOME_RETENTION,
                                                    DEVELOPMENT_COST_SHAPE)
            environments[scenario] = environment
            reference_states = build_no_action_reference(config, income_retention_fraction=INCOME_RETENTION)
            reward_functions[scenario] = make_reward_function(settings["reward_mode"], reference_states)

        action_count = len(next(iter(environments.values())).action_names)
        checkpoints, history = train_agents_across_scenarios(
            environments, action_count, settings["training_seed"], checkpoint, encoder, decode_actions,
            reward_functions, LEARNING_RATE, DISCOUNT_FACTOR, EPSILON_START, settings["epsilon_decay"],
            EPSILON_MINIMUM, lambda row, total: show_training_progress(row, total),
        )
        agents = checkpoints[episodes]

    print()
    return agents, history


def train_dqn_model(settings):
    from study_3_pollutant_dynamics.dqn_training import train_dqn_agents

    environments = {}
    reward_functions = {}

    for scenario in settings["training_scenarios"]:
        config, environment = _make_environment(scenario, settings["episode_years"], 1.0,
                                                settings["income_retention_fraction"],
                                                settings["development_cost_shape"])
        environments[scenario] = environment
        reference_states = build_no_action_reference(config,
                                                     income_retention_fraction=settings["income_retention_fraction"])
        reward_functions[scenario] = make_reward_function(settings["reward_mode"], reference_states)

    training_options = {"learning_rate": settings["learning_rate"],
                        "discount_factor": settings["discount_factor"],
                        "epsilon_start": settings["epsilon_start"],
                        "epsilon_decay": settings["epsilon_decay"],
                        "epsilon_minimum": settings["epsilon_minimum"],
                        "replay_buffer_size": settings["replay_buffer_size"],
                        "batch_size": settings["batch_size"],
                        "learning_starts": settings["learning_starts"],
                        "train_frequency": settings["train_frequency"],
                        "target_update_frequency": settings["target_update_frequency"],
                        "hidden_size": settings["hidden_size"],
                        "progress_callback": show_training_progress}
    agents, history = train_dqn_agents(environments, reward_functions, settings["training_seed"],
                                       settings["training_episodes"], **training_options)
    print()
    return agents, history


def train_ppo_model(settings):
    from study_3_pollutant_dynamics.ppo_training import train_ppo_agents

    environments = {}
    reward_functions = {}
    for scenario in settings["training_scenarios"]:
        config, environment = _make_environment(scenario, settings["episode_years"], 1.0,
                                                settings["income_retention_fraction"],
                                                settings["development_cost_shape"])
        environments[scenario] = environment
        reference_states = build_no_action_reference(config,
                                                     income_retention_fraction=settings["income_retention_fraction"])
        reward_functions[scenario] = make_reward_function(settings["reward_mode"], reference_states)

    training_options = {key: settings[key] for key in PPO_SETTINGS}
    training_options["progress_callback"] = show_training_progress
    agents, history = train_ppo_agents(environments, reward_functions, settings["training_seed"],
                                       settings["training_episodes"], **training_options)
    print()
    return agents, history


def save_training_run(settings, agents, history):
    os.makedirs(RUNS_DIRECTORY, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = timestamp + "_" + settings["reward_mode"] + "_" + str(settings["training_seed"])
    run_directory = RUNS_DIRECTORY + "/" + run_name

    suffix = 1

    while os.path.exists(run_directory):
        run_name = timestamp + "_" + settings["reward_mode"] + "_" + str(settings["training_seed"]) + "_" + str(suffix)
        run_directory = RUNS_DIRECTORY + "/" + run_name
        suffix += 1

    model_directory = run_directory + "/q_tables"

    if settings["algorithm"] == "independent_dqn":
        model_directory = run_directory + "/dqn_models"
    elif settings["algorithm"] == "independent_ppo":
        model_directory = run_directory + "/ppo_models"

    os.makedirs(model_directory, exist_ok=True)
    settings["run_name"] = run_name
    settings["run_directory"] = run_directory
    settings["model_directory"] = model_directory
    settings["training_state_counts"] = []

    for region, agent in enumerate(agents):
        if settings["algorithm"] in ("independent_dqn", "independent_ppo"):
            state_count = len(agent.training_states) if settings["algorithm"] == "independent_dqn" else agent.training_state_count
        else:
            state_count = len(agent.q_values)
        settings["training_state_counts"].append(state_count)
        filename = model_directory + "/region_" + str(region)

        if settings["algorithm"] in ("independent_dqn", "independent_ppo"):
            agent.save(filename + ".pt")
        else:
            save_q_table(agent, filename + ".json")

    with open(run_directory + "/settings.json", "w", encoding="utf-8") as output_file:
        json.dump(settings, output_file, indent=2)

    with open(run_directory + "/training_history.json", "w", encoding="utf-8") as output_file:
        json.dump(history, output_file)

    return run_name, run_directory


def summarize_training(settings, agents, history, run_directory):
    recent_rows = history[-min(50, len(history)):]
    recent_reward = 0.0

    if recent_rows:
        for row in recent_rows:
            recent_reward += row.get("mean_cumulative_reward_per_region", 0.0)
        recent_reward /= len(recent_rows)

    state_counts = []

    for region, agent in enumerate(agents):
        if settings["algorithm"] in ("independent_dqn", "independent_ppo"):
            state_counts.append(len(agent.training_states))
        else:
            state_counts.append(len(agent.q_values))

    print("\nTraining finished.")
    print("Algorithm:", settings["algorithm"])
    print("Reward:", settings["reward_mode"])
    print("Training conditions:", ", ".join(settings["training_scenarios"]))
    print("Episodes:", settings["training_episodes"])
    print("Mean reward per region over the final 50 episodes:", round(recent_reward, 3))
    print("Learned states by region:", state_counts)
    print("Saved model and settings:", run_directory)


def load_saved_model(settings):
    agents = []
    model_directory = settings["model_directory"]

    if settings["algorithm"] == "independent_dqn":
        from study_3_pollutant_dynamics.dqn_agent import DQNAgent
    elif settings["algorithm"] == "independent_ppo":
        from study_3_pollutant_dynamics.ppo_agent import PPOAgent

    for region in range(4):
        if settings["algorithm"] == "independent_dqn":
            agent = DQNAgent(len(STATE_BIN_EDGES), len(ACTION_NAMES), settings["training_seed"] * 10 + region,
                             settings["learning_rate"], settings["discount_factor"], 0.0,
                             settings["replay_buffer_size"], settings["batch_size"], settings["learning_starts"],
                             settings["train_frequency"], settings["target_update_frequency"], settings["hidden_size"])
            agent.load(model_directory + "/region_" + str(region) + ".pt")
            agent.training_state_count = settings["training_state_counts"][region]
        elif settings["algorithm"] == "independent_ppo":
            agent = PPOAgent(len(STATE_BIN_EDGES), len(ACTION_NAMES), settings["training_seed"] * 10 + region,
                             settings["learning_rate"], settings["discount_factor"], settings["gae_lambda"],
                             settings["clip_ratio"], settings["entropy_coefficient"], settings["value_coefficient"],
                             settings["update_epochs"], settings["minibatch_size"], settings["hidden_size"])
            agent.load(model_directory + "/region_" + str(region) + ".pt")
        else:
            agent = TabularQLearner(len(ACTION_NAMES), settings["training_seed"] * 10 + region,
                                    settings["learning_rate"], settings["discount_factor"], 0.0)
            load_q_table(agent, model_directory + "/region_" + str(region) + ".json")
        agents.append(agent)

    return agents


def evaluate_saved_model(run_directory, settings):
    agents = load_saved_model(settings)
    encoder = make_state_encoder(STATE_BIN_EDGES)
    dqn_evaluator = None

    if settings["algorithm"] == "independent_dqn":
        from study_3_pollutant_dynamics.dqn_training import evaluate_dqn_agents
        dqn_evaluator = evaluate_dqn_agents
    elif settings["algorithm"] == "independent_ppo":
        from study_3_pollutant_dynamics.ppo_training import evaluate_ppo_agents
        ppo_evaluator = evaluate_ppo_agents
    scenarios = choose_option("Evaluate under which pollution conditions? ",
                              {"1": "Low", "2": "Reference", "3": "High", "4": "All three"}, "4")
    selected_scenarios = list(POLLUTION_SCENARIOS)

    if scenarios != "4":
        selected_scenarios = [POLLUTION_SCENARIOS[int(scenarios) - 1]]

    evaluation_rows = []

    for scenario in selected_scenarios:
        config, environment = _make_environment(scenario, settings["episode_years"], 1.0,
                                                settings["income_retention_fraction"],
                                                settings["development_cost_shape"])
        reference_states = build_no_action_reference(config,
                                                     income_retention_fraction=settings["income_retention_fraction"])
        reward_function = make_reward_function(settings["reward_mode"], reference_states)
        if settings["algorithm"] == "independent_dqn":
            learned = dqn_evaluator(environment, agents, reward_function)
        elif settings["algorithm"] == "independent_ppo":
            learned = ppo_evaluator(environment, agents, reward_function)
        else:
            learned = evaluate_agent_policy(environment, agents, encoder, decode_actions, reward_function)
        learned_actions = learned.pop("action_counts_by_region")
        learned.pop("trajectory")
        learned["policy"] = "learned_" + settings["algorithm"].replace("independent_", "")
        learned["evaluation_scenario"] = scenario
        evaluation_rows.append(learned)
        _, baseline_environment = _make_environment(scenario, settings["episode_years"], 1.0,
                                                    settings["income_retention_fraction"],
                                                    settings["development_cost_shape"])
        baseline_trajectory, terminated, truncated = run_fixed_policy_episode("balanced", baseline_environment, 0)
        baseline = summarize_episode_metrics(baseline_trajectory, baseline_environment)
        baseline["policy"] = "balanced_fixed"
        baseline["evaluation_scenario"] = scenario
        evaluation_rows.append(baseline)
        print("\nScenario:", scenario)
        print("Policy | Final EDI | Net income | Final currency | Mitigation | CO2 | PM2.5 | Nitrate-N | Lead")

        for row in (learned, baseline):
            summary_line = "{0} | {1:.2f} | {2:.1f} | {3:.1f} | {4:.1f} | {5:.2f} | {6:.3f} | {7:.3f} | {8:.3f}".format(
                row["policy"], row["final_mean_EDI"], row["cumulative_net_income"], row["final_mean_currency"],
                row["mitigation_spending"], row["final_CO2_ppm"], row["final_mean_PM2.5_ug_m3"],
                row["final_mean_nitrate_mg_N_L"], row["final_mean_lead_total_kg"])
            print(summary_line)

        total_action_counts = {}

        for region_counts in learned_actions:
            for action, count in region_counts.items():
                total_action_counts[action] = total_action_counts.get(action, 0) + count

        print("Learned action counts:", total_action_counts)

    output_path = run_directory + "/evaluation_summary.json"

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(evaluation_rows, output_file, indent=2)

    print("Saved evaluation summary:", output_path)


def list_saved_runs():
    if not os.path.isdir(RUNS_DIRECTORY):
        return []

    runs = []

    for name in os.listdir(RUNS_DIRECTORY):
        run_directory = RUNS_DIRECTORY + "/" + name
        settings_file = run_directory + "/settings.json"

        if os.path.isdir(run_directory) and os.path.isfile(settings_file):
            runs.append(name)

    runs.sort()
    return runs


def evaluate_existing_run():
    runs = list_saved_runs()

    if not runs:
        print("No saved training runs were found in " + RUNS_DIRECTORY)
        return

    run_choices = {}

    for index, run_name in enumerate(runs, start=1):
        run_choices[str(index)] = run_name

    selected = choose_option("Saved training runs: ", run_choices)
    run_name = run_choices[selected]
    run_directory = RUNS_DIRECTORY + "/" + run_name

    with open(run_directory + "/settings.json", "r", encoding="utf-8") as input_file:
        settings = json.load(input_file)

    evaluate_saved_model(run_directory, settings)


def train_and_offer_evaluation():
    settings = choose_training_settings()

    if settings["algorithm"] in ("independent_dqn", "independent_ppo"):
        try:
            import torch
        except ImportError:
            print("DQN and PPO require PyTorch. Install it with: python -m pip install -e \".[rl]\"")
            return

    print("\nStarting " + settings["algorithm"].replace("_", " ") + ".")
    agents, history = train_selected_model(settings)
    run_name, run_directory = save_training_run(settings, agents, history)
    summarize_training(settings, agents, history, run_directory)
    answer = input("Evaluate this saved model now? [Y/n] ").strip().lower()

    if answer in ("", "y", "yes"):
        evaluate_saved_model(run_directory, settings)


def main():
    print("Study 3 reinforcement learning")
    choice = choose_option("Main menu: ", {"1": "Train a model ", "2": "Evaluate a saved model ", "0": "Exit"}, "1 ")

    if choice == "1":
        train_and_offer_evaluation()
    elif choice == "2":
        evaluate_existing_run()
    else:
        print("Goodbye.")


if __name__ == "__main__":
    main()
