"""Coupled annual environment for Study 3 baseline policy evaluation."""
import copy

from study_3_pollutant_dynamics.damage_parameters import DAMAGE_SCENARIOS
from study_3_pollutant_dynamics.development_costs import (DEVELOPMENT_COST_SHAPES, calculate_development_cost)
from study_3_pollutant_dynamics.economic_parameters import (PM25_REFERENCE_EMISSIONS_TONS, REGIONAL_ECONOMIC_PARAMETERS,
                                                            REGIONAL_EMISSION_PROFILE_MULTIPLIERS, STAGE_2_SCENARIOS)
from study_3_pollutant_dynamics.environmental_damage import (apply_income_damage, calculate_pollutant_damage_fractions,
                                                             combine_damage_fractions, update_lagged_exposure)
from study_3_pollutant_dynamics.mitigation import (cleanup_capacity, lead_mobile_fraction, prevention_fraction, purchase_costs)
from study_3_pollutant_dynamics.mitigation_parameters import (DEVELOPMENT_COST_PER_EDI_POINT, MITIGATION_POLICIES,
                                                              PROJECT_PURCHASE_FREQUENCY_YEARS)
from study_3_pollutant_dynamics.regional_economics import (calculate_annual_income, calculate_emission_activity_index)
from study_3_pollutant_dynamics.environment import PollutantDynamicsEnvironment


INTEGRATED_ACTIONS = {
    "maintain": {"development_points": 0.0, "investment": None},
    "develop": {"development_points": 1.0, "investment": None},
    "prevention": {"development_points": 0.0, "investment": "prevention_focused"},
    "develop_prevention": {"development_points": 1.0, "investment": "prevention_focused"},
    "cleanup": {"development_points": 0.0, "investment": "cleanup_focused"},
    "develop_cleanup": {"development_points": 1.0, "investment": "cleanup_focused"},
    "balanced": {"development_points": 0.0, "investment": "balanced"},
    "develop_balanced": {"development_points": 1.0, "investment": "balanced"},
}


class Study3IntegratedEnvironment:
    """Couple Study 1 pollutant processes with the tested economy and policies.

    Actions are one named choice per region per year. No reward function is
    defined in this environment; this stage evaluates physical and economic
    baselines before selecting an RL objective.
    """

    def __init__(self, config, starting_currency_multiplier=1.0, development_cost_multiplier=1.0, investment_cost_multiplier=1.0,
                 development_cost_shape="constant", income_damage_multiplier=1.0, income_damage_persistence=1.0,
                 income_retention_fraction=1.0):

        if starting_currency_multiplier < 0:
            raise ValueError("starting_currency_multiplier cannot be negative")
        if development_cost_multiplier < 0 or investment_cost_multiplier < 0:
            raise ValueError("Cost multipliers cannot be negative")
        if income_damage_multiplier < 0:
            raise ValueError("income_damage_multiplier cannot be negative")
        if not 0 <= income_damage_persistence <= 1:
            raise ValueError("income_damage_persistence must be between zero and one")
        if not 0 <= income_retention_fraction <= 1:
            raise ValueError("income_retention_fraction must be between zero and one")

        if development_cost_shape not in DEVELOPMENT_COST_SHAPES:
            raise ValueError(f"Unknown development cost shape: {development_cost_shape}")

        if len(config.regions) != len(REGIONAL_ECONOMIC_PARAMETERS):
            raise ValueError("Economic parameters must match configured regions")

        self.config = copy.deepcopy(config)
        self.pollutant_environment = PollutantDynamicsEnvironment(self.config)
        self.regions = self.config.regions
        self.action_names = tuple(INTEGRATED_ACTIONS)
        self.starting_currency_multiplier = starting_currency_multiplier
        self.development_cost_multiplier = development_cost_multiplier
        self.investment_cost_multiplier = investment_cost_multiplier
        self.development_cost_shape = development_cost_shape
        self.income_damage_multiplier = income_damage_multiplier
        self.income_damage_persistence = income_damage_persistence
        self.income_retention_fraction = income_retention_fraction
        configured_mobile_fractions = self.config.lead["mobile_fraction"]
        self.base_lead_mobile_fractions = []

        for region_index in range(len(self.regions)):
            mobile_fraction = configured_mobile_fractions

            if isinstance(configured_mobile_fractions, (list, tuple)):
                mobile_fraction = configured_mobile_fractions[region_index]
            self.base_lead_mobile_fractions.append(mobile_fraction)

        self.base_nitrate_inputs = tuple(self.config.nitrate["annual_input_kg_n"])
        self.base_lead_inputs = tuple(self.config.lead["annual_input_kg_pb"])
        self.step_count = 0
        self.started = False
        self.done = False
        self.edi = []
        self.currency = []
        self.previous_income_base = []
        self.previous_edi_income_factor = []
        self.projects = []
        self.next_project_purchase_year = []
        self.lagged_lead_exposure = []
        self.lead_reference_exposure = []
        self.lagged_co2_exposure = 0.0
        self.investment_spending = []
        self.development_spending = []
        self.unaffordable_development_attempts = []
        self.unaffordable_investment_attempts = []

    def reset(self, seed=0):
        self.config.lead["mobile_fraction"] = list(self.base_lead_mobile_fractions)
        observations = self.pollutant_environment.reset(seed)
        self.step_count = 0
        self.done = False
        self.started = True
        self.edi = []
        self.currency = []
        self.previous_income_base = []
        self.previous_edi_income_factor = []
        self.projects = []
        self.next_project_purchase_year = []
        self.lagged_lead_exposure = []
        self.lead_reference_exposure = []
        self.investment_spending = []
        self.development_spending = []
        self.unaffordable_development_attempts = []
        self.unaffordable_investment_attempts = []

        for region_index, region_parameters in enumerate(REGIONAL_ECONOMIC_PARAMETERS):
            self.edi.append(region_parameters["initial_edi"])
            starting_currency = (region_parameters["initial_currency"] * self.starting_currency_multiplier)
            self.currency.append(starting_currency)
            self.previous_income_base.append(region_parameters["annual_income"])
            self.previous_edi_income_factor.append(1.0)
            self.projects.append([])
            self.next_project_purchase_year.append(1)
            self.investment_spending.append(0.0)
            self.development_spending.append(0.0)
            self.unaffordable_development_attempts.append(0)
            self.unaffordable_investment_attempts.append(0)

            lead_mobile_kg = self.pollutant_environment.mobile_stocks["Lead"][region_index]
            lead_mobile_mg_kg = (lead_mobile_kg / self.config.lead["soil_mass_kg"] * 1000000)
            self.lagged_lead_exposure.append(lead_mobile_mg_kg)
            self.lead_reference_exposure.append(lead_mobile_mg_kg)

        self.lagged_co2_exposure = self.config.co2["initial_concentration_ppm"]
        return self._observations(observations)

    def _observations(self, pollutant_observations=None):
        if pollutant_observations is None:
            pollutant_observations = self.pollutant_environment._observations()

        observations = []
        for region_index, pollutant_observation in enumerate(pollutant_observations):
            observation = dict(pollutant_observation)
            observation["EDI"] = self.edi[region_index]
            observation["currency"] = self.currency[region_index]
            observation["region_index"] = region_index
            observation["year"] = self.step_count
            observations.append(observation)

        return tuple(observations)

    def _check_actions(self, actions):
        if not isinstance(actions, (list, tuple)):
            raise ValueError("actions must contain one named action per region")

        if len(actions) != len(self.regions):
            raise ValueError("Provide one action for each region")

        for action in actions:
            if not isinstance(action, str) or action not in INTEGRATED_ACTIONS:
                raise ValueError(f"Unknown Study 3 action: {action}")

    def _regional_emissions(self, year, activity_factors):
        emission_inputs = {"pm25": [], "co2": [], "nitrate": [], "lead": []}
        mobile_fractions = []
        cleanup_capacities = {"CO2": [], "Nitrate-N": [], "Lead": []}
        prevention_fractions = {"PM2.5": [], "CO2": [], "Nitrate-N": [], "Lead": []}

        for region_index in range(len(self.regions)):
            region_projects = self.projects[region_index]
            for pollutant in prevention_fractions:
                fraction = prevention_fraction(region_projects, year, pollutant)
                prevention_fractions[pollutant].append(fraction)

            pm25_reference_tons = (PM25_REFERENCE_EMISSIONS_TONS * REGIONAL_EMISSION_PROFILE_MULTIPLIERS["PM2.5"][region_index])
            pm25_emissions_tons = pm25_reference_tons
            pm25_emissions_tons *= activity_factors[region_index]
            pm25_emissions_tons *= 1 - prevention_fractions["PM2.5"][region_index]
            pm25_input_tons = pm25_emissions_tons - pm25_reference_tons
            emission_inputs["pm25"].append(pm25_input_tons)

            co2_input = self.config.co2["reference_emissions_gtco2_year"][region_index]
            co2_input *= activity_factors[region_index]
            co2_input *= 1 - prevention_fractions["CO2"][region_index]
            co2_removal = cleanup_capacity(region_projects, year, "CO2",
                                           self.config.co2["reference_emissions_gtco2_year"][region_index],0.0)
            emission_inputs["co2"].append(co2_input - co2_removal)
            cleanup_capacities["CO2"].append(co2_removal)

            nitrate_input = self.base_nitrate_inputs[region_index]
            nitrate_input *= REGIONAL_EMISSION_PROFILE_MULTIPLIERS["Nitrate-N"][region_index]
            nitrate_input *= activity_factors[region_index]
            nitrate_input *= 1 - prevention_fractions["Nitrate-N"][region_index]
            emission_inputs["nitrate"].append(nitrate_input)
            nitrate_cleanup = cleanup_capacity(region_projects, year, "Nitrate-N", 0.0,
                                               self.config.nitrate["initial_stock_kg_n"][region_index])
            cleanup_capacities["Nitrate-N"].append(nitrate_cleanup)

            lead_input = self.base_lead_inputs[region_index]
            lead_input *= REGIONAL_EMISSION_PROFILE_MULTIPLIERS["Lead"][region_index]
            lead_input *= activity_factors[region_index]
            lead_input *= 1 - prevention_fractions["Lead"][region_index]
            emission_inputs["lead"].append(lead_input)
            lead_cleanup = cleanup_capacity(region_projects, year, "Lead", 0.0,
                                            self.config.lead["initial_stock_kg_pb"][region_index])
            cleanup_capacities["Lead"].append(lead_cleanup)
            mobile_fractions.append(lead_mobile_fraction(region_projects, year, self.base_lead_mobile_fractions[region_index]))

        self.config.pm25["annual_emission_change_tons"] = emission_inputs["pm25"]
        self.config.co2["annual_emissions_gtco2_year"] = emission_inputs["co2"]
        self.config.nitrate["annual_input_kg_n"] = emission_inputs["nitrate"]
        self.config.lead["annual_input_kg_pb"] = emission_inputs["lead"]
        self.config.lead["mobile_fraction"] = mobile_fractions

        return emission_inputs, cleanup_capacities, prevention_fractions

    def step(self, actions):
        if not self.started or self.done:
            raise RuntimeError("Reset the environment before stepping")

        self._check_actions(actions)
        year = self.step_count + 1
        gross_incomes = []
        income_factors = []
        activity_factors = []
        development_gains = []
        development_costs = []
        investment_costs = []
        investment_packages = []
        package_purchased = []
        currency_before_actions = []
        development_requested_costs = []
        investment_requested_costs = []
        development_was_requested = []
        development_was_affordable = []
        investment_was_due = []
        investment_was_affordable = []
        simultaneous_affordability = []

        for region_index, action_name in enumerate(actions):
            action = INTEGRATED_ACTIONS[action_name]
            region_parameters = REGIONAL_ECONOMIC_PARAMETERS[region_index]
            current_edi = self.edi[region_index]
            starting_balance = self.currency[region_index]
            currency_before_actions.append(starting_balance)
            activity_factor = calculate_emission_activity_index(1.0, current_edi, region_parameters["initial_edi"],
                                                                STAGE_2_SCENARIOS["emission_edi_elasticity"]["reference"])
            income_factor = calculate_annual_income(1.0, 0.0, 0, current_edi,
                                                    region_parameters["initial_edi"],
                                                    STAGE_2_SCENARIOS["income_edi_elasticity"]["reference"])
            activity_factors.append(activity_factor)
            income_factors.append(income_factor)
            gross_income = self.previous_income_base[region_index]
            gross_income *= 1 + region_parameters["annual_income_growth_rate"]
            gross_income *= income_factor / self.previous_edi_income_factor[region_index]
            gross_incomes.append(gross_income)

            development_gain = action["development_points"]

            if current_edi >= 100:
                development_gain = 0.0

            development_gain = min(development_gain, 100 - current_edi)
            development_is_requested = development_gain > 0
            requested_development_cost = calculate_development_cost(current_edi, region_parameters["initial_edi"],
                                                                    development_gain, self.development_cost_shape,
                                                                    DEVELOPMENT_COST_PER_EDI_POINT)

            requested_development_cost *= self.development_cost_multiplier
            development_is_affordable = starting_balance >= requested_development_cost
            development_cost = requested_development_cost

            if self.currency[region_index] < development_cost:
                if development_cost > 0:
                    self.unaffordable_development_attempts[region_index] += 1
                development_gain = 0.0
                development_cost = 0.0

            self.currency[region_index] -= development_cost
            self.development_spending[region_index] += development_cost
            development_gains.append(development_gain)
            development_costs.append(development_cost)
            development_requested_costs.append(requested_development_cost)
            development_was_requested.append(development_is_requested)
            development_was_affordable.append(development_is_affordable)

            package_name = action["investment"]
            investment_package = ()
            package_cost = 0.0
            requested_package_cost = 0.0
            bought_package = False
            package_is_due = False
            package_is_affordable = False

            if package_name is not None:
                investment_package = MITIGATION_POLICIES[package_name]
                if year >= self.next_project_purchase_year[region_index]:
                    package_is_due = True
                    requested_package_cost = (purchase_costs(investment_package) * self.investment_cost_multiplier)
                    package_cost = requested_package_cost
                    if self.currency[region_index] >= package_cost:
                        package_is_affordable = True
                        self.currency[region_index] -= package_cost
                        self.investment_spending[region_index] += package_cost

                        for category, pollutant in investment_package:
                            project = {"category": category, "pollutant": pollutant, "investment_year": year}
                            self.projects[region_index].append(project)
                        self.next_project_purchase_year[region_index] = (year + PROJECT_PURCHASE_FREQUENCY_YEARS)
                        bought_package = True
                    else:
                        self.unaffordable_investment_attempts[region_index] += 1
                else:
                    package_cost = 0.0
            if not bought_package:
                package_cost = 0.0

            investment_costs.append(package_cost)
            investment_requested_costs.append(requested_package_cost)
            investment_was_due.append(package_is_due)
            investment_was_affordable.append(package_is_affordable)
            simultaneous_affordability.append(development_is_requested and package_is_due
                                              and starting_balance >= requested_development_cost + requested_package_cost)
            investment_packages.append(package_name)
            package_purchased.append(bought_package)

        emission_inputs, cleanup_capacities, prevention_fractions = (self._regional_emissions(year, activity_factors))
        nitrate_before = tuple(self.pollutant_environment.stocks["Nitrate-N"])
        lead_before = tuple(self.pollutant_environment.stocks["Lead"])
        pollutant_observations, terminated, truncated, info = (self.pollutant_environment.step())

        nitrate_removed = []
        lead_removed = []
        lead_mobile_after = []
        next_nitrate_stocks = list(info["stocks"]["Nitrate-N"])
        next_lead_stocks = list(info["stocks"]["Lead"])

        for region_index in range(len(self.regions)):
            nitrate_mass = info["stocks"]["Nitrate-N"][region_index]
            nitrate_cleanup = min(nitrate_mass, cleanup_capacities["Nitrate-N"][region_index])
            nitrate_removed.append(nitrate_cleanup)
            nitrate_mass -= nitrate_cleanup
            next_nitrate_stocks[region_index] = nitrate_mass
            lead_mass = info["stocks"]["Lead"][region_index]
            lead_cleanup = min(lead_mass, cleanup_capacities["Lead"][region_index])
            lead_removed.append(lead_cleanup)
            lead_mass -= lead_cleanup
            mobile_lead = lead_mass * self.config.lead["mobile_fraction"][region_index]
            lead_mobile_after.append(mobile_lead)

            next_lead_stocks[region_index] = lead_mass

        nitrate_final = tuple(next_nitrate_stocks)
        lead_final = tuple(next_lead_stocks)
        info["stocks"]["Nitrate-N"] = nitrate_final
        info["stocks"]["Lead"] = lead_final
        self.pollutant_environment.stocks["Nitrate-N"] = nitrate_final
        self.pollutant_environment.mobile_stocks["Nitrate-N"] = nitrate_final
        self.pollutant_environment.stocks["Lead"] = lead_final
        self.pollutant_environment.mobile_stocks["Lead"] = tuple(lead_mobile_after)
        info["mobile_stocks"]["Nitrate-N"] = nitrate_final
        info["mobile_stocks"]["Lead"] = tuple(lead_mobile_after)
        info["cleanup_removed"] = {"Nitrate-N_kg": tuple(nitrate_removed), "Lead_kg": tuple(lead_removed),
                                   "CO2_Gt_year": tuple(cleanup_capacities["CO2"])}

        damage_settings = DAMAGE_SCENARIOS["reference"]
        damage_settings = dict(damage_settings)
        damage_rate_names = ("pm25_rate_per_ug_m3", "co2_rate_per_ppm", "nitrate_relative_rate", "lead_relative_rate")

        for rate_name in damage_rate_names:
            damage_settings[rate_name] *= self.income_damage_multiplier

        lagged_co2 = update_lagged_exposure(self.lagged_co2_exposure, info["stocks"]["CO2"], damage_settings["co2_delay_years"])
        self.lagged_co2_exposure = lagged_co2
        regional_results = []

        for region_index, region_parameters in enumerate(REGIONAL_ECONOMIC_PARAMETERS):
            nitrate_concentration = (1000 * nitrate_final[region_index] / self.config.nitrate["water_volume_m3"])
            lead_mobile_concentration = (lead_mobile_after[region_index] / self.config.lead["soil_mass_kg"] * 1000000)
            self.lagged_lead_exposure[region_index] = update_lagged_exposure(self.lagged_lead_exposure[region_index],
                                                                             lead_mobile_concentration,
                                                                             damage_settings["lead_delay_years"])
            damage_fractions = calculate_pollutant_damage_fractions(info["stocks"]["PM2.5"][region_index],
                                                                    self.config.pm25["baseline_exposure_ug_m3"][region_index],
                                                                    lagged_co2, self.config.co2["initial_concentration_ppm"],
                                                                    nitrate_concentration, self.config.nitrate["initial_concentration_mg_n_l"],
                                                                    self.lagged_lead_exposure[region_index],
                                                                    self.lead_reference_exposure[region_index], damage_settings)

            total_damage_fraction = combine_damage_fractions(damage_fractions)
            net_income = apply_income_damage(gross_incomes[region_index], total_damage_fraction)
            retained_income = net_income * self.income_retention_fraction
            recurring_expense = net_income - retained_income
            self.currency[region_index] += retained_income
            next_income_base = gross_income * (1 - self.income_damage_persistence * total_damage_fraction)
            self.previous_income_base[region_index] = next_income_base
            self.previous_edi_income_factor[region_index] = income_factors[region_index]
            self.edi[region_index] += development_gains[region_index]
            self.edi[region_index] = min(100.0, self.edi[region_index])

            regional_prevention_fractions = {}
            for pollutant in prevention_fractions:
                regional_prevention_fractions[pollutant] = (prevention_fractions[pollutant][region_index])

            regional_results.append({
                "region": self.regions[region_index],
                "action": actions[region_index],
                "year": year,
                "EDI": self.edi[region_index],
                "currency": self.currency[region_index],
                "gross_income": gross_incomes[region_index],
                "net_income": net_income,
                "retained_income": retained_income,
                "recurring_expense": recurring_expense,
                "damage_fraction": total_damage_fraction,
                "next_year_income_base": next_income_base,
                "damage_by_pollutant": damage_fractions,
                "development_cost": development_costs[region_index],
                "development_requested_cost": development_requested_costs[region_index],
                "development_requested": development_was_requested[region_index],
                "development_affordable": development_was_affordable[region_index],
                "investment_cost": investment_costs[region_index],
                "investment_requested_cost": investment_requested_costs[region_index],
                "investment_due": investment_was_due[region_index],
                "investment_affordable": investment_was_affordable[region_index],
                "currency_before_actions": currency_before_actions[region_index],
                "simultaneous_affordability": simultaneous_affordability[region_index],
                "investment_package": investment_packages[region_index],
                "package_purchased": package_purchased[region_index],
                "emission_activity_index": activity_factors[region_index],
                "prevention_fractions": regional_prevention_fractions,
                "Nitrate_cleanup_kg": nitrate_removed[region_index],
                "Lead_cleanup_kg": lead_removed[region_index],
                "Lead_mobile_kg": lead_mobile_after[region_index],
            })

        self.step_count = year
        truncated = self.step_count >= self.config.episode_steps
        self.done = terminated or truncated
        info["regional_economy"] = tuple(regional_results)
        info["emission_inputs"] = emission_inputs
        info["mass_balance_residuals"] = self._mass_balance_residuals(
            nitrate_before,
            lead_before,
            emission_inputs,
            info,
            nitrate_removed,
            lead_removed,
        )
        info["unaffordable_development_attempts"] = tuple(self.unaffordable_development_attempts)
        info["unaffordable_investment_attempts"] = tuple(self.unaffordable_investment_attempts)

        return self._observations(), terminated, truncated, info

    def _mass_balance_residuals(self, nitrate_before, lead_before, inputs, info, nitrate_removed, lead_removed):
        nitrate_residuals = []
        lead_residuals = []

        for region_index in range(len(self.regions)):
            nitrate_flow = info["nitrate_flows"][region_index]
            nitrate_residual = nitrate_before[region_index]
            nitrate_residual += inputs["nitrate"][region_index]
            nitrate_residual -= nitrate_flow["outflow_kg_n"]
            nitrate_residual -= nitrate_flow["denitrified_kg_n"]
            nitrate_residual -= nitrate_removed[region_index]
            nitrate_residual -= info["stocks"]["Nitrate-N"][region_index]
            nitrate_residuals.append(nitrate_residual)

            lead_flow = info["lead_flows"][region_index]
            lead_residual = lead_before[region_index]
            lead_residual += inputs["lead"][region_index]
            lead_residual -= lead_flow["exported_kg_pb"]
            lead_residual -= lead_removed[region_index]
            lead_residual -= info["stocks"]["Lead"][region_index]
            lead_residuals.append(lead_residual)

        return {"Nitrate-N_kg": tuple(nitrate_residuals), "Lead_kg": tuple(lead_residuals)}
