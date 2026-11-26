"""State encoding and reward functions for the first Study 3 RL baseline."""

from study_3_pollutant_dynamics.integrated_environment import INTEGRATED_ACTIONS, Study3IntegratedEnvironment


ACTION_NAMES = tuple(INTEGRATED_ACTIONS)
STATE_BIN_EDGES = {
    "EDI": (60, 70, 80, 90, 95, 98),
    "currency": (100, 300, 600, 1200, 2400),
    "PM2.5_ug_m3": (9, 15),
    "CO2_ppm": (450, 500, 550),
    "Nitrate-N_mg_L": (5, 10),
    "Lead_total_kg": (15, 25),
    "Lead_mobile_kg": (0.005, 0.02),
    "year": (10, 20, 30),
}
POLLUTION_REWARD_SCALES = {"PM2.5_ug_m3": 9.0, "CO2_ppm": 100.0, "Nitrate-N_mg_L": 10.0}
POLLUTION_WEIGHT = 1.0
MIXED_LOCAL_WEIGHT = 0.7


def _value_bin(value, boundaries):
    bin_index = 0

    for boundary in boundaries:
        if value >= boundary:
            bin_index += 1

    return bin_index


def encode_observation(observation, bin_edges=None):
    if bin_edges is None:
        bin_edges = STATE_BIN_EDGES
    state = []

    for feature_name, boundaries in bin_edges.items():
        state.append(_value_bin(observation[feature_name], boundaries))

    return tuple(state)


def decode_actions(action_indices):
    actions = []

    for action_index in action_indices:
        if type(action_index) is not int or action_index < 0 or action_index >= len(ACTION_NAMES):
            raise ValueError("Action index is outside the Study 3 action space")
        actions.append(ACTION_NAMES[action_index])

    return actions


def build_no_action_reference(config, starting_currency_multiplier=1.0, income_retention_fraction=0.6):
    environment = Study3IntegratedEnvironment(config, starting_currency_multiplier=starting_currency_multiplier,
                                              development_cost_shape="constant",
                                              income_retention_fraction=income_retention_fraction)
    observations = environment.reset(0)
    reference_states = [observations]

    while not environment.done:
        observations, terminated, truncated, info = environment.step(["maintain"] * len(environment.regions))
        reference_states.append(observations)

    return reference_states


def calculate_reward_components(observations, next_observations, info, reference_states, pollution_weight=POLLUTION_WEIGHT):
    reference_observations = reference_states[info["step"]]
    regional_components = []
    local_rewards = []

    for region_index, (observation, next_observation) in enumerate(zip(observations, next_observations)):
        reference = reference_observations[region_index]
        region_result = info["regional_economy"][region_index]
        pm25_change = (next_observation["PM2.5_ug_m3"] - reference["PM2.5_ug_m3"]) / POLLUTION_REWARD_SCALES["PM2.5_ug_m3"]
        co2_change = (next_observation["CO2_ppm"] - reference["CO2_ppm"]) / POLLUTION_REWARD_SCALES["CO2_ppm"]
        nitrate_change = (next_observation["Nitrate-N_mg_L"] - reference["Nitrate-N_mg_L"]) / POLLUTION_REWARD_SCALES["Nitrate-N_mg_L"]
        initial_mobile_lead = reference_states[0][region_index]["Lead_mobile_kg"]
        lead_change = (next_observation["Lead_mobile_kg"] - reference["Lead_mobile_kg"]) / max(initial_mobile_lead, 1e-12)
        pollution_pressure = (pm25_change + co2_change + nitrate_change + lead_change) / 4
        development_reward = next_observation["EDI"] - observation["EDI"]
        pollution_reward = -pollution_weight * pollution_pressure
        local_reward = development_reward + pollution_reward
        local_rewards.append(local_reward)
        regional_components.append({
            "region": region_result["region"],
            "development_reward": development_reward,
            "pollution_pressure": pollution_pressure,
            "pollution_reward": pollution_reward,
            "local_reward": local_reward,
            "net_income": region_result["net_income"],
            "development_spending": region_result["development_cost"],
            "mitigation_spending": region_result["investment_cost"],
            "damage_fraction": region_result["damage_fraction"],
            "PM2.5_normalized_change": pm25_change,
            "CO2_normalized_change": co2_change,
            "nitrate_normalized_change": nitrate_change,
            "lead_normalized_change": lead_change,
        })

    global_reward = sum(local_rewards) / len(local_rewards)

    return {"regional_components": regional_components, "local_rewards": local_rewards, "global_reward": global_reward}


def rewards_for_mode(reward_mode, observations, next_observations, info, reference_states, pollution_weight=POLLUTION_WEIGHT):
    components = calculate_reward_components(observations, next_observations, info, reference_states, pollution_weight)

    if reward_mode == "local":
        return components["local_rewards"]

    if reward_mode == "global":
        return [components["global_reward"]] * len(components["local_rewards"])

    if reward_mode == "mixed":
        rewards = []

        for local_reward in components["local_rewards"]:
            mixed_reward = MIXED_LOCAL_WEIGHT * local_reward + (1 - MIXED_LOCAL_WEIGHT) * components["global_reward"]
            rewards.append(mixed_reward)

        return rewards

    raise ValueError(f"Unknown reward mode: {reward_mode}")


def make_reward_function(reward_mode, reference_states, pollution_weight=POLLUTION_WEIGHT):
    def reward_function(observations, actions, info, next_observations):
        return rewards_for_mode(reward_mode, observations, next_observations, info, reference_states, pollution_weight)

    return reward_function
