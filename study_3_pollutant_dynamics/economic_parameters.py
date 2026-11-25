"""Regional economy inputs, kept separate from pollutant dynamics."""
REGIONAL_ECONOMIC_PARAMETERS = (
    {
        "region": "Northeast",
        "initial_edi": 70.0,
        "initial_currency": 700.0,
        "annual_income": 65.0,
        "annual_income_growth_rate": 0.015,
    },
    {
        "region": "Midwest",
        "initial_edi": 60.0,
        "initial_currency": 550.0,
        "annual_income": 55.0,
        "annual_income_growth_rate": 0.012,
    },
    {
        "region": "South",
        "initial_edi": 55.0,
        "initial_currency": 450.0,
        "annual_income": 50.0,
        "annual_income_growth_rate": 0.020,
    },
    {
        "region": "West",
        "initial_edi": 75.0,
        "initial_currency": 800.0,
        "annual_income": 75.0,
        "annual_income_growth_rate": 0.018,
    },
)


# These are the regional multipliers proposed earlier for exploration.
REGIONAL_EMISSION_PROFILE_MULTIPLIERS = {"PM2.5": (0.9, 1.2, 1.1, 0.9), "CO2": (0.9, 1.2, 1.1, 1.0),
                                         "Nitrate-N": (0.8, 1.4, 1.2, 0.8), "Lead": (0.9, 1.3, 1.0, 0.8)}


# Synthetic reference source mass used only to convert economic activity to PM2.5 emissions changes for the
# source-receptor approximation. The exposure baselines are already present in the initial state, so only deviations from
# each region's starting activity are passed into that equation.
PM25_REFERENCE_EMISSIONS_TONS = 5000.0

STAGE_2_SCENARIOS = {
    "income_edi_elasticity": {"low": 0.0, "reference": 0.5, "high": 1.0},
    "emission_edi_elasticity": {"low": 0.4, "reference": 0.8, "high": 1.2}
}
