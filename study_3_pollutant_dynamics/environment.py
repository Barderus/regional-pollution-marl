"""Annual environmental dynamics for Study 3, without policy actions or rewards."""
import random

from study_3_pollutant_dynamics.pollutant_models import (co2_concentration, lead_annual_step, nitrate_annual_step, pm25_exposure)

class PollutantDynamicsEnvironment:
    def __init__(self, config):
        self.config = config
        self.action_names = ("none",)
        self.rng = random.Random()
        self.step_count = 0
        self.started = False
        self.done = False
        self.stocks = {}
        self.mobile_stocks = {}
        self.co2_emission_cohorts = []

    def _regional_lead_mobile_fraction(self, region):
        mobile_fraction = self.config.lead["mobile_fraction"]

        if isinstance(mobile_fraction, (list, tuple)):
            return mobile_fraction[region]

        return mobile_fraction

    def reset(self, seed=0):
        self.rng.seed(seed)
        self.step_count = 0
        self.done = False
        self.started = True
        self.co2_emission_cohorts = []

        nitrate_stocks = []
        lead_stocks = []
        lead_mobile_stocks = []

        for region in range(len(self.config.regions)):
            nitrate_stocks.append(self.config.nitrate["initial_stock_kg_n"][region])
            lead_stock = self.config.lead["initial_stock_kg_pb"][region]
            lead_stocks.append(lead_stock)
            lead_mobile_stocks.append(lead_stock * self._regional_lead_mobile_fraction(region))

        self.stocks = {"PM2.5": tuple(self.config.pm25["baseline_exposure_ug_m3"]),
                       "CO2": self.config.co2["initial_concentration_ppm"], "Nitrate-N": tuple(nitrate_stocks),
                       "Lead": tuple(lead_stocks)
                       }

        self.mobile_stocks = {"Nitrate-N": tuple(nitrate_stocks), "Lead": tuple(lead_mobile_stocks)}

        return self._observations()

    def _observations(self):
        observations = []

        for region in range(len(self.config.regions)):
            nitrate_concentration = (1000 * self.stocks["Nitrate-N"][region] / self.config.nitrate["water_volume_m3"])
            observation = {"PM2.5_ug_m3": self.stocks["PM2.5"][region], "CO2_ppm": self.stocks["CO2"],
                           "Nitrate-N_mg_L": nitrate_concentration, "Nitrate-N_stock_kg": self.stocks["Nitrate-N"][region],
                           "Lead_total_kg": self.stocks["Lead"][region], "Lead_mobile_kg": self.mobile_stocks["Lead"][region]}
            observations.append(observation)

        return tuple(observations)

    def decode_actions(self, action_indices):
        decoded_actions = []

        for action_index in action_indices:
            if type(action_index) is not int or action_index != 0:
                raise ValueError("The pollutant dynamics stage only supports the no-op action")
            decoded_actions.append("none")

        return decoded_actions

    def _check_actions(self, regional_actions):
        if regional_actions is None:
            return

        if not isinstance(regional_actions, (list, tuple)):
            raise ValueError("regional_actions must be a list or tuple of no-op actions")

        if len(regional_actions) != len(self.config.regions):
            raise ValueError("Provide one no-op action per region")

        for action in regional_actions:
            if action not in (None, "none", 0):
                raise ValueError("Policy actions are not part of the pollutant dynamics stage")

    def step(self, regional_actions=None):
        if not self.started or self.done:
            raise RuntimeError("Reset the environment before stepping")

        self._check_actions(regional_actions)

        pm25_exposure_values = pm25_exposure(self.config.pm25["baseline_exposure_ug_m3"],
                                             self.config.pm25["annual_emission_change_tons"],
                                             self.config.pm25["transfer_ug_m3_per_1000_tons"],
                                             self.config.pm25["cross_region_share"]
                                             )
        co2_deviation_gtco2 = 0.0
        for region in range(len(self.config.regions)):
            co2_deviation_gtco2 += (self.config.co2["annual_emissions_gtco2_year"][region]
                                    - self.config.co2["reference_emissions_gtco2_year"][region])

        aging_cohorts = []
        for age_years, mass_gtco2 in self.co2_emission_cohorts:
            aging_cohorts.append((age_years + 1, mass_gtco2))
        aging_cohorts.append((0, co2_deviation_gtco2))

        self.co2_emission_cohorts = aging_cohorts
        next_year = self.step_count + 1
        co2_ppm = co2_concentration(self.config.co2["initial_concentration_ppm"], self.config.co2["background_growth_ppm_year"],
                                    next_year, self.co2_emission_cohorts, self.config.co2["ppm_per_gtco2"])

        next_nitrate_stocks = []
        nitrate_flows = []
        next_nitrate_mobile = []
        next_lead_stocks = []
        next_lead_mobile = []
        lead_flows = []

        for region in range(len(self.config.regions)):
            nitrate_result = nitrate_annual_step(self.stocks["Nitrate-N"][region], self.config.nitrate["annual_discharge_m3"],
                                                 self.config.nitrate["water_volume_m3"],
                                                 self.config.nitrate["annual_input_kg_n"][region],
                                                 self.config.nitrate["denitrification_rate_per_year"])

            next_nitrate_stocks.append(nitrate_result["stock_kg_n"])
            next_nitrate_mobile.append(nitrate_result["stock_kg_n"])
            nitrate_flows.append(nitrate_result)

            lead_result = lead_annual_step(self.stocks["Lead"][region], self.config.lead["annual_input_kg_pb"][region],
                                           self._regional_lead_mobile_fraction(region),
                                           self.config.lead["annual_mobile_export_fraction"])

            next_lead_stocks.append(lead_result["total_stock_kg_pb"])
            next_lead_mobile.append(lead_result["mobile_stock_kg_pb"])
            lead_flows.append(lead_result)

        self.stocks = {"PM2.5": pm25_exposure_values, "CO2": co2_ppm, "Nitrate-N": tuple(next_nitrate_stocks),
                       "Lead": tuple(next_lead_stocks)}

        self.mobile_stocks = {"Nitrate-N": tuple(next_nitrate_mobile), "Lead": tuple(next_lead_mobile)}

        self.step_count = next_year
        truncated = self.step_count >= self.config.episode_steps
        terminated = False
        self.done = terminated or truncated

        info = {
            "step": self.step_count,
            "time_step_name": self.config.time_step_name,
            "stocks": self.stocks,
            "mobile_stocks": self.mobile_stocks,
            "gross_emissions": {
                "PM2.5_change_tons": self.config.pm25["annual_emission_change_tons"],
                "CO2_gtco2_year": self.config.co2["annual_emissions_gtco2_year"],
                "Nitrate-N_kg_year": self.config.nitrate["annual_input_kg_n"],
                "Lead_kg_year": self.config.lead["annual_input_kg_pb"],
            },
            "co2_emission_deviation_gtco2": co2_deviation_gtco2,
            "nitrate_flows": tuple(nitrate_flows),
            "lead_flows": tuple(lead_flows),
            "failed_pollutants": (),
        }

        return self._observations(), terminated, truncated, info
