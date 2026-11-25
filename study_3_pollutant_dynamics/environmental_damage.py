"""Pollutant-specific, synthetic economic damage calculations"""
import math

def update_lagged_exposure(previous_value, current_value, delay_years):
    if delay_years < 0:
        raise ValueError("delay_years cannot be negative")

    if delay_years == 0:
        return current_value

    response_fraction = 1 - math.exp(-1 / delay_years)
    lagged_value = previous_value + response_fraction * (current_value - previous_value)

    return lagged_value


def calculate_pollutant_damage_fractions(pm25_exposure, pm25_reference, co2_lagged_ppm, co2_reference_ppm,
                                         nitrate_mg_n_l, nitrate_reference_mg_n_l, lead_mobile_mg_kg,
                                         lead_reference_mobile_mg_kg, damage_settings):
    pollutant_values = (
        pm25_exposure,
        pm25_reference,
        co2_lagged_ppm,
        co2_reference_ppm,
        nitrate_mg_n_l,
        nitrate_reference_mg_n_l,
        lead_mobile_mg_kg,
        lead_reference_mobile_mg_kg,
    )

    for value in pollutant_values:
        if value < 0:
            raise ValueError("Pollutant concentrations cannot be negative")

    damage_rates = (damage_settings["pm25_rate_per_ug_m3"], damage_settings["co2_rate_per_ppm"],
                    damage_settings["nitrate_relative_rate"], damage_settings["lead_relative_rate"])

    for rate in damage_rates:
        if rate < 0:
            raise ValueError("Damage rates cannot be negative")

    if pm25_reference < 0 or co2_reference_ppm < 0:
        raise ValueError("Pollutant reference values cannot be negative")

    if nitrate_reference_mg_n_l <= 0 or lead_reference_mobile_mg_kg <= 0:
        raise ValueError("Nitrate and lead reference values must be positive")

    pm25_excess = max(0.0, pm25_exposure - pm25_reference)
    co2_excess = max(0.0, co2_lagged_ppm - co2_reference_ppm)
    nitrate_relative_excess = max(0.0, nitrate_mg_n_l / nitrate_reference_mg_n_l - 1)
    lead_relative_excess = max(0.0, lead_mobile_mg_kg / lead_reference_mobile_mg_kg - 1)

    pm25_exponent = damage_settings["pm25_rate_per_ug_m3"] * pm25_excess
    co2_exponent = damage_settings["co2_rate_per_ppm"] * co2_excess
    nitrate_exponent = damage_settings["nitrate_relative_rate"] * nitrate_relative_excess ** 2
    lead_exponent = damage_settings["lead_relative_rate"] * lead_relative_excess

    damage_fractions = {"PM2.5": 1 - math.exp(-pm25_exponent), "CO2": 1 - math.exp(-co2_exponent),
                        "Nitrate-N": 1 - math.exp(-nitrate_exponent), "Lead": 1 - math.exp(-lead_exponent)}

    return damage_fractions


def combine_damage_fractions(damage_fractions):
    remaining_income_fraction = 1.0

    for damage_fraction in damage_fractions.values():
        if not 0 <= damage_fraction < 1:
            raise ValueError("Each damage fraction must be between zero and one")
        remaining_income_fraction *= 1 - damage_fraction

    total_damage_fraction = 1 - remaining_income_fraction

    return total_damage_fraction


def apply_income_damage(gross_income, total_damage_fraction):
    if gross_income < 0:
        raise ValueError("gross_income cannot be negative")

    if not 0 <= total_damage_fraction < 1:
        raise ValueError("total_damage_fraction must be between zero and one")

    net_income = gross_income * (1 - total_damage_fraction)

    return net_income
