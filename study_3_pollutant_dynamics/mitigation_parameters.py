"""Synthetic investment costs, effects, delays, and persistence."""

PROJECT_PURCHASE_FREQUENCY_YEARS = 5
DEVELOPMENT_COST_PER_EDI_POINT = 25.0
PREVENTION_FRACTION_CAP = 0.75


MITIGATION_PROJECTS = {
    "prevention": {
        "PM2.5": { "cost": 15.0, "emissions_reduction_fraction": 0.25, "delay_years": 1, "persistence_years": 5},
        "CO2": {"cost": 40.0, "emissions_reduction_fraction": 0.15, "delay_years": 1, "persistence_years": 5},
        "Nitrate-N": {"cost": 15.0, "emissions_reduction_fraction": 0.20, "delay_years": 1, "persistence_years": 5},
        "Lead": {"cost": 10.0, "emissions_reduction_fraction": 0.25, "delay_years": 1, "persistence_years": 5},
    },
    "cleanup": {
        "CO2": {"cost": 45.0, "capacity_fraction_reference_emissions": 0.005, "delay_years": 3, "persistence_years": 10},
        "Nitrate-N": {"cost": 25.0, "capacity_fraction_initial_stock": 0.05, "delay_years": 2, "persistence_years": 5},
        "Lead": {"cost": 30.0, "capacity_fraction_initial_stock": 0.005, "delay_years": 3, "persistence_years": 10},
    },
    "containment": {
        "Lead": {"cost": 20.0, "mobile_fraction_multiplier": 0.50, "delay_years": 2, "persistence_years": 10},
    },
}


MITIGATION_POLICIES = {
    "no_development_no_investment": (),
    "development_only": (),
    "prevention_focused": (("prevention", "PM2.5"), ("prevention", "CO2"), ("prevention", "Nitrate-N"), ("prevention", "Lead")),
    "cleanup_focused": (("cleanup", "CO2"), ("cleanup", "Nitrate-N"), ("cleanup", "Lead")),
    "balanced": (("prevention", "PM2.5"), ("prevention", "CO2"), ("prevention", "Nitrate-N"), ("prevention", "Lead"),
                 ("cleanup", "CO2"), ("cleanup", "Nitrate-N"), ("cleanup", "Lead"), ("containment", "Lead")),
}
