"""Evaluate trained regional agents on held-out environment seeds."""

from study_2_economic_sustainability.environment import EconomicSustainabilityEnvironment


def run_episode(environment, agents, seed, action_modifier=None):
    observations = environment.reset(seed)
    transitions = []
    terminated = False
    truncated = False

    while True:
        development_before = environment.development_indices
        currency_before = environment.currency_balances
        pollution_before = environment.pollution_loads
        pollutant_loads_before = {}

        for pollutant in environment.config.pollutants:
            pollutant_loads_before[pollutant] = environment.pollutant_loads[pollutant]

        chosen_actions = []

        for region in environment.config.regions:
            action = agents[region].select_action(
                observations[region],
                greedy=True,
            )
            chosen_actions.append(action)

        actions = tuple(chosen_actions)

        if action_modifier is not None:
            actions = action_modifier(actions)

        next_observations, rewards, terminated, truncated, info = environment.step(actions)

        transition = {
            "step": environment.step_count,
            "observations": observations,
            "development_before": development_before,
            "currency_before": currency_before,
            "pollution_before": pollution_before,
            "pollutant_loads_before": pollutant_loads_before,
            "chosen_actions": tuple(chosen_actions),
            "actions": tuple(actions),
            "rewards": rewards,
            "next_observations": next_observations,
            "info": info,
            "terminated": terminated,
            "truncated": truncated,
        }
        transitions.append(transition)
        observations = next_observations

        if terminated or truncated:
            break

    return transitions, terminated, truncated


def evaluate_episode(environment, agents, seed):
    pollutant_totals = {}
    pollutant_maxima = {}
    gpi_total = 0.0
    prevention_spending = 0.0
    cleanup_spending = 0.0

    for pollutant in environment.config.pollutants:
        pollutant_totals[pollutant] = 0.0
        pollutant_maxima[pollutant] = 0.0

    transitions, terminated, truncated = run_episode(environment, agents, seed)

    for transition in transitions:
        info = transition["info"]
        gpi_total += info["global_pollution_index"]
        prevention_spending += sum(info["prevention_costs_paid"])
        cleanup_spending += sum(info["cleanup_costs_paid"])

        for pollutant in environment.config.pollutants:
            for region in environment.config.regions:
                load = info["pollutant_loads"][pollutant][region]
                pollutant_totals[pollutant] += load
                pollutant_maxima[pollutant] = max(
                    pollutant_maxima[pollutant], load
                )

    region_count = len(environment.config.regions)
    row = {
        "seed": seed,
        "steps": environment.step_count,
        "failure": terminated,
        "mean_gpi": gpi_total / environment.step_count,
        "final_mean_edi": sum(environment.development_indices) / region_count,
        "final_currency": sum(environment.currency_balances) / region_count,
        "prevention_spending": prevention_spending,
        "cleanup_spending": cleanup_spending,
    }

    for pollutant in environment.config.pollutants:
        row[f"mean_{pollutant}"] = (
            pollutant_totals[pollutant] / (environment.step_count * region_count)
        )
        row[f"maximum_{pollutant}"] = pollutant_maxima[pollutant]

    return row


def evaluate_trained_policies(trained_policies, environment_config, evaluation_seeds):
    evaluation_rows = []

    for (reward_mode, training_seed), agents in trained_policies.items():
        environment = EconomicSustainabilityEnvironment(
            reward_mode,
            environment_config,
        )

        for evaluation_seed in evaluation_seeds:
            row = evaluate_episode(environment, agents, evaluation_seed)
            row["reward_mode"] = reward_mode
            row["training_seed"] = training_seed
            evaluation_rows.append(row)

    return evaluation_rows
