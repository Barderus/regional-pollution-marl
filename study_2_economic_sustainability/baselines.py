BASELINE_NAMES = ("save", "develop", "mitigate", "coordinate", "random_affordable")


def choose_baseline_actions(name, environment, observations, rng=None):
    # Choose actions from current observations and affordable costs
    if name not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline: {name}")

    costs = environment.config.action_costs
    actions = []

    for region in range(4):
        balance = environment.currency_balances[region]
        edi = environment.development_indices[region]

        if name == "save":
            action = 0

        elif name == "develop":
            action = 1 if edi < 100 and balance >= costs[1] else 0

        elif name == "mitigate":
            action = 0
            for option in (4, 3, 2):
                if balance >= costs[option]:
                    action = option
                    break

        elif name == "coordinate":
            # Use only the local pollution bin and shared GPI bin to decide when pollution needs attention. Exact balance is used to avoid an unaffordable action.
            local_pollution_bin = observations[region][2]
            global_pollution_bin = observations[region][3]

            if local_pollution_bin >= 1 or global_pollution_bin >= 1:
                action = 0
                for option in (4, 3, 2):
                    if balance >= costs[option]:
                        action = option
                        break
            elif edi < 70 and balance >= costs[1]:
                action = 1
            else:
                action = 0

        else:
            if rng is None:
                raise ValueError("random_affordable requires a seeded random generator")
            affordable = []
            for option in range(5):
                if balance >= costs[option] and (option != 1 or edi < 100):
                    affordable.append(option)
            action = rng.choice(affordable)

        actions.append(action)

    return tuple(actions)
