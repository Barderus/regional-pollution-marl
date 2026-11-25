"""Validated configuration for the annual Study 3 pollutant model."""

class EnvironmentConfig:
    def __init__(self, parameters):
        self.regions = tuple(parameters["regions"])
        self.episode_steps = parameters["episode_steps"]
        self.time_step_name = "year"
        self.scenario_names = parameters["scenario_names"]
        self.pollutants = ("PM2.5", "CO2", "Nitrate-N", "Lead")
        self.local_pollutants = ("PM2.5", "Nitrate-N", "Lead")
        self.global_pollutant = "CO2"
        self.pm25 = parameters["pm25"]
        self.co2 = parameters["co2"]
        self.nitrate = parameters["nitrate"]
        self.lead = parameters["lead"]

        self._check_parameters()

    def _check_parameters(self):
        if len(self.regions) < 2:
            raise ValueError("At least two regions are required")

        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")

        regional_fields = (
            ("PM2.5 baseline exposure", self.pm25["baseline_exposure_ug_m3"]),
            ("PM2.5 annual emission changes", self.pm25["annual_emission_change_tons"]),
            ("CO2 reference emissions", self.co2["reference_emissions_gtco2_year"]),
            ("CO2 annual emissions", self.co2["annual_emissions_gtco2_year"]),
            ("Nitrate annual input loads", self.nitrate["annual_input_kg_n"]),
            ("Lead annual input loads", self.lead["annual_input_kg_pb"]),
        )

        for field_name, values in regional_fields:
            if len(values) != len(self.regions):
                raise ValueError(f"{field_name} must have one value per region")

        for value in self.pm25["baseline_exposure_ug_m3"]:
            if value < 0:
                raise ValueError("PM2.5 baseline concentrations cannot be negative")

        if self.pm25["transfer_ug_m3_per_1000_tons"] < 0:
            raise ValueError("PM2.5 transfer coefficient cannot be negative")

        if not 0 <= self.pm25["cross_region_share"] <= 1:
            raise ValueError("PM2.5 cross-region share must be between zero and one")

        if self.co2["initial_concentration_ppm"] <= 0:
            raise ValueError("Initial CO2 concentration must be positive")

        if self.co2["ppm_per_gtco2"] <= 0:
            raise ValueError("CO2 mass conversion must be positive")

        if self.nitrate["water_volume_m3"] <= 0:
            raise ValueError("Nitrate water volume must be positive")

        if self.nitrate["annual_discharge_m3"] < 0:
            raise ValueError("Nitrate annual discharge cannot be negative")

        if self.nitrate["denitrification_rate_per_year"] < 0:
            raise ValueError("Nitrate removal rate cannot be negative")

        if self.lead["soil_mass_kg"] <= 0:
            raise ValueError("Lead soil mass must be positive")

        if self.lead["kd_l_per_kg"] <= 0:
            raise ValueError("Lead Kd must be positive")

        if self.lead["water_to_soil_ratio_l_per_kg"] < 0:
            raise ValueError("Lead water-to-soil ratio cannot be negative")

        if not 0 <= self.lead["annual_mobile_export_fraction"] <= 1:
            raise ValueError("Lead mobile export fraction must be between zero and one")

        mobile_fractions = self.lead["mobile_fraction"]
        if isinstance(mobile_fractions, (list, tuple)):
            if len(mobile_fractions) != len(self.regions):
                raise ValueError("Lead mobile fractions must have one value per region")
            for mobile_fraction in mobile_fractions:
                if not 0 <= mobile_fraction <= 1:
                    raise ValueError("Lead mobile fractions must be between zero and one")
        elif not 0 <= mobile_fractions <= 1:
            raise ValueError("Lead mobile fraction must be between zero and one")

        fraction_names = ("pm25", "co2", "nitrate", "lead")
        for scenario_name in fraction_names:
            if scenario_name not in self.scenario_names:
                raise ValueError(f"Missing {scenario_name} scenario label")
