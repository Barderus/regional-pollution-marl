"""Standalone regional income and relative emissions equations."""
def _development_factor(current_edi, initial_edi, elasticity):
    if not 0 <= current_edi <= 100:
        raise ValueError("current_edi must be between 0 and 100")

    if not 0 <= initial_edi <= 100:
        raise ValueError("initial_edi must be between 0 and 100")

    if elasticity < 0:
        raise ValueError("elasticity cannot be negative")

    factor = ((1 + current_edi) / (1 + initial_edi)) ** elasticity
    return factor


def calculate_annual_income(base_income, annual_growth_rate, year_number, current_edi, initial_edi, edi_income_elasticity):
    if base_income < 0:
        raise ValueError("base_income cannot be negative")

    if annual_growth_rate <= -1:
        raise ValueError("annual_growth_rate must be greater than -1")

    if year_number < 0:
        raise ValueError("year_number cannot be negative")

    development_factor = _development_factor(current_edi, initial_edi, edi_income_elasticity)
    annual_income = base_income * ((1 + annual_growth_rate) ** year_number)
    annual_income *= development_factor

    return annual_income


def calculate_emission_activity_index(profile_multiplier, current_edi, initial_edi, edi_emission_elasticity):
    if profile_multiplier < 0:
        raise ValueError("profile_multiplier cannot be negative")

    development_factor = _development_factor(current_edi, initial_edi, edi_emission_elasticity)
    emission_activity_index = profile_multiplier * development_factor

    return emission_activity_index
