"""Synthetic Stage 3 environmental-damage scenarios."""

# These values are exploratory coefficients, not estimated economic damages.
# Delay values are annual time constants for the synthetic response lag.
DAMAGE_SCENARIOS = {
    "low": {
        "pm25_rate_per_ug_m3": 0.001,
        "co2_rate_per_ppm": 0.00005,
        "nitrate_relative_rate": 0.10,
        "lead_relative_rate": 0.01,
        "co2_delay_years": 15.0,
        "lead_delay_years": 8.0,
    },
    "reference": {
        "pm25_rate_per_ug_m3": 0.002,
        "co2_rate_per_ppm": 0.00010,
        "nitrate_relative_rate": 0.25,
        "lead_relative_rate": 0.03,
        "co2_delay_years": 10.0,
        "lead_delay_years": 5.0,
    },
    "high": {
        "pm25_rate_per_ug_m3": 0.004,
        "co2_rate_per_ppm": 0.00020,
        "nitrate_relative_rate": 0.50,
        "lead_relative_rate": 0.06,
        "co2_delay_years": 5.0,
        "lead_delay_years": 2.0,
    },
}
