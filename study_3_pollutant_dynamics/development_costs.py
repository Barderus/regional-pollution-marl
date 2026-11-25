"""Illustrative marginal development-cost curves for Study 3."""

DEVELOPMENT_COST_SHAPES = ("constant", "linear", "quadratic")

def calculate_development_cost(current_edi, initial_edi, edi_points, cost_shape="constant", cost_per_point=25.0):
    if not 0 <= current_edi <= 100:
        raise ValueError("current_edi must be between 0 and 100")

    if not 0 <= initial_edi < 100:
        raise ValueError("initial_edi must be between 0 and 100")

    if edi_points < 0:
        raise ValueError("edi_points cannot be negative")

    if cost_per_point < 0:
        raise ValueError("cost_per_point cannot be negative")

    if cost_shape not in DEVELOPMENT_COST_SHAPES:
        raise ValueError(f"Unknown development cost shape: {cost_shape}")

    progress = (current_edi - initial_edi) / (100 - initial_edi)
    cost_factor = 1.0

    if cost_shape == "linear":
        cost_factor = 1 + progress
    elif cost_shape == "quadratic":
        cost_factor = (1 + progress) ** 2

    development_cost = cost_per_point * cost_factor * edi_points

    return development_cost
