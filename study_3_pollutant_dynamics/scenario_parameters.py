"""Editable source anchors and low/reference/high scenario assumptions."""

import copy
REGION_NAMES = ("Northeast", "Midwest", "South", "West")

PM25_BASELINE_EXPOSURE_UG_M3 = {"Northeast": 7.420, "Midwest": 8.079, "South": 8.145, "West": 10.608}

# Approximate 2023 state totals summed into Census regions from EIA Table CO2.1. Units are Gt CO2/year.
# This territorial energy inventory is a reference path, not a second addition to the observed global atmospheric concentration trend.
CO2_REFERENCE_EMISSIONS_GTCO2_YEAR = {"Northeast": 0.5956, "Midwest": 1.1551, "South": 2.1270, "West": 0.9074}

# The Keokuk USGS pilot covers 217 days in 2023. Its flow-weighted mean is 2.15 mg N/L; annualized mean discharge is a scenario anchor, not a full-year mean.
KEOKUK_FLOW_WEIGHTED_NITRATE_MG_N_L = 2.15
KEOKUK_ANNUALIZED_DISCHARGE_M3 = 68116564179.5

SCENARIO_SETTINGS = {
    "pm25": {
        "low": {"transfer_ug_m3_per_1000_tons": 0.005, "cross_region_share": 0.05},
        "reference": {"transfer_ug_m3_per_1000_tons": 0.020, "cross_region_share": 0.20},
        "high": {"transfer_ug_m3_per_1000_tons": 0.080, "cross_region_share": 0.40},
    },
    "co2": {
        "low": {"background_growth_ppm_year": 1.94, "emission_multiplier": 0.90},
        "reference": {"background_growth_ppm_year": 2.50, "emission_multiplier": 1.00},
        "high": {"background_growth_ppm_year": 3.76, "emission_multiplier": 1.10},
    },
    "nitrate": {
        "low": {"input_multiplier": 0.80, "denitrification_rate_per_year": 0.30, "residence_time_years": 0.25},
        "reference": {"input_multiplier": 1.00, "denitrification_rate_per_year": 0.10, "residence_time_years": 0.50},
        "high": {"input_multiplier": 1.20, "denitrification_rate_per_year": 0.02, "residence_time_years": 1.00},
    },
    "lead": {
        "low": {"soil_concentration_mg_kg": 13.5, "kd_l_per_kg": 44580.0, "annual_input_kg_pb": 0.01, "annual_mobile_export_fraction": 0.05},
        "reference": {"soil_concentration_mg_kg": 18.1, "kd_l_per_kg": 2587.0, "annual_input_kg_pb": 0.10, "annual_mobile_export_fraction": 0.01},
        "high": {"soil_concentration_mg_kg": 23.9, "kd_l_per_kg": 150.0, "annual_input_kg_pb": 1.00, "annual_mobile_export_fraction": 0.001},
    },
}


def get_parameters(scenarios=None, episode_steps=40):
    selected_scenarios = {"pm25": "reference", "co2": "reference", "nitrate": "reference", "lead": "reference"}

    if scenarios is not None:
        for pollutant in selected_scenarios:
            if pollutant in scenarios:
                selected_scenarios[pollutant] = scenarios[pollutant]

    settings = {}
    for pollutant, scenario_name in selected_scenarios.items():
        if scenario_name not in SCENARIO_SETTINGS[pollutant]:
            raise ValueError(f"Unknown {pollutant} scenario: {scenario_name}")
        settings[pollutant] = copy.deepcopy(SCENARIO_SETTINGS[pollutant][scenario_name])

    regions = list(REGION_NAMES)
    pm25_baseline = []
    co2_reference = []

    for region in regions:
        pm25_baseline.append(PM25_BASELINE_EXPOSURE_UG_M3[region])
        co2_reference.append(CO2_REFERENCE_EMISSIONS_GTCO2_YEAR[region])

    co2_emissions = []
    for emission in co2_reference:
        co2_emissions.append(emission * settings["co2"]["emission_multiplier"])

    discharge_m3_year = KEOKUK_ANNUALIZED_DISCHARGE_M3
    water_volume_m3 = discharge_m3_year * settings["nitrate"]["residence_time_years"]
    nitrate_initial_concentration = KEOKUK_FLOW_WEIGHTED_NITRATE_MG_N_L
    nitrate_initial_stock = nitrate_initial_concentration * water_volume_m3 / 1000
    nitrate_reference_input = nitrate_initial_concentration * (discharge_m3_year+ settings["nitrate"]["denitrification_rate_per_year"]
                                                               * water_volume_m3) / 1000
    nitrate_input = nitrate_reference_input * settings["nitrate"]["input_multiplier"]

    soil_area_m2 = 10000.0
    soil_depth_m = 0.05
    bulk_density_kg_m3 = 1350.0
    soil_mass_kg = soil_area_m2 * soil_depth_m * bulk_density_kg_m3
    lead_initial_stock = settings["lead"]["soil_concentration_mg_kg"] * soil_mass_kg / 1000000
    water_to_soil_ratio_l_per_kg = 1.0
    lead_mobile_fraction = water_to_soil_ratio_l_per_kg / (water_to_soil_ratio_l_per_kg + settings["lead"]["kd_l_per_kg"])

    nitrate_input_values = []
    lead_input_values = []
    for region in regions:
        nitrate_input_values.append(nitrate_input)
        lead_input_values.append(settings["lead"]["annual_input_kg_pb"])

    parameters = {
        "regions": regions,
        "episode_steps": episode_steps,
        "scenario_names": selected_scenarios,
        "pm25": {
            "baseline_exposure_ug_m3": pm25_baseline,
            "annual_emission_change_tons": [0.0, 0.0, 0.0, 0.0],
            "transfer_ug_m3_per_1000_tons": settings["pm25"]["transfer_ug_m3_per_1000_tons"],
            "cross_region_share": settings["pm25"]["cross_region_share"],
        },
        "co2": {
            "initial_concentration_ppm": 425.62,
            "background_growth_ppm_year": settings["co2"]["background_growth_ppm_year"],
            "ppm_per_gtco2": 1 / 7.78,
            "reference_emissions_gtco2_year": co2_reference,
            "annual_emissions_gtco2_year": co2_emissions,
        },
        "nitrate": {
            "initial_stock_kg_n": [nitrate_initial_stock] * 4,
            "initial_concentration_mg_n_l": nitrate_initial_concentration,
            "water_volume_m3": water_volume_m3,
            "annual_discharge_m3": discharge_m3_year,
            "denitrification_rate_per_year": settings["nitrate"]["denitrification_rate_per_year"],
            "annual_input_kg_n": nitrate_input_values,
        },
        "lead": {
            "initial_stock_kg_pb": [lead_initial_stock] * 4,
            "initial_concentration_mg_kg": settings["lead"]["soil_concentration_mg_kg"],
            "soil_mass_kg": soil_mass_kg,
            "kd_l_per_kg": settings["lead"]["kd_l_per_kg"],
            "water_to_soil_ratio_l_per_kg": water_to_soil_ratio_l_per_kg,
            "mobile_fraction": lead_mobile_fraction,
            "annual_mobile_export_fraction": settings["lead"]["annual_mobile_export_fraction"],
            "annual_input_kg_pb": lead_input_values,
        },
    }

    return parameters
