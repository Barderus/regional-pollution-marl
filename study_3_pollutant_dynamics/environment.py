"""Pollutant stock accounting for Study 3's first model stage."""

import random
import math


# Joos et al. (2013) multi-model mean CO2 impulse response coefficients.
CO2_RESPONSE_COMPONENTS = (
    (0.2173, None),
    (0.2240, 394.4),
    (0.2824, 36.54),
    (0.2763, 4.304),
)


class PollutantDynamicsEnvironment:
    def __init__(self, config):
        self.config = config
        self.action_names = ("none",) + tuple(config.interventions)
        self.rng = random.Random()
        self.stocks = {}
        self.immobilized_stocks = {}
        self.global_emission_cohorts = []
        self.step_count = 0
        self.started = False
        self.done = False

    def reset(self, seed=0):
        self.rng.seed(seed)
        self.stocks = {}

        for pollutant in self.config.local_pollutants:
            initial_values = self.config.initial_stocks[pollutant]
            self.stocks[pollutant] = tuple(initial_values)
            self.immobilized_stocks[pollutant] = tuple(
                self.config.initial_immobilized_stocks[pollutant]
            )

        initial_global_stock = float(
            self.config.initial_stocks[self.config.global_pollutant]
        )
        self.global_emission_cohorts = [(0, initial_global_stock)]
        self.stocks[self.config.global_pollutant] = initial_global_stock
        self.step_count = 0
        self.started = True
        self.done = False

        return self._observations()

    def _observations(self):
        observations = []

        for region in self.config.regions:
            region_state = {}

            for pollutant in self.config.local_pollutants:
                region_state[pollutant] = self.stocks[pollutant][region]
                region_state[f"{pollutant}_mobile"] = (
                    self.stocks[pollutant][region]
                    - self.immobilized_stocks[pollutant][region]
                )

            region_state[self.config.global_pollutant] = self.stocks[
                self.config.global_pollutant
            ]
            observations.append(region_state)

        return tuple(observations)

    def _co2_response_fraction(self, age_years):
        remaining_fraction = 0.0

        for coefficient, time_constant in CO2_RESPONSE_COMPONENTS:
            if time_constant is None:
                remaining_fraction += coefficient
            else:
                remaining_fraction += coefficient * math.exp(
                    -age_years / time_constant
                )

        return remaining_fraction

    def _global_stock_from_cohorts(self, cohorts):
        stock = 0.0

        for age_years, mass in cohorts:
            stock += mass * self._co2_response_fraction(age_years)

        return stock

    def _intervention(self, action):
        if action is None or action == "none":
            return {
                "prevention_fraction": {},
                "cleanup_capacity": {},
                "immobilization_fraction": {},
                "global_removal": 0.0,
            }

        if action not in self.config.interventions:
            raise ValueError(f"Unknown intervention: {action}")

        return self.config.interventions[action]

    def decode_actions(self, action_indices):
        decoded_actions = []

        for action_index in action_indices:
            if type(action_index) is not int:
                raise ValueError("Action indices must be integers")

            if action_index < 0 or action_index >= len(self.action_names):
                raise ValueError("Action index is outside the configured action list")

            decoded_actions.append(self.action_names[action_index])

        return decoded_actions

    def step(self, regional_actions):
        if not self.started or self.done:
            raise RuntimeError("Reset the environment before stepping")

        if not isinstance(regional_actions, (list, tuple)):
            raise ValueError("regional_actions must be a list or tuple")

        if len(regional_actions) != len(self.config.regions):
            raise ValueError("Provide one intervention name per region")

        interventions = []

        for action in regional_actions:
            interventions.append(self._intervention(action))

        emissions_after_prevention = {}

        for pollutant in self.config.local_pollutants:
            emissions_after_prevention[pollutant] = []

            for region in self.config.regions:
                gross_emissions = self.config.annual_emissions[pollutant][region]
                prevention_fraction = interventions[region][
                    "prevention_fraction"
                ].get(pollutant, 0.0)
                prevention_fraction = min(max(prevention_fraction, 0.0), 1.0)
                emissions_after_prevention[pollutant].append(
                    gross_emissions * (1.0 - prevention_fraction)
                )

        global_pollutant = self.config.global_pollutant
        global_emissions = sum(self.config.annual_emissions[global_pollutant])
        global_prevention_fraction = 0.0

        for region, intervention in enumerate(interventions):
            prevention = intervention["prevention_fraction"]
            regional_fraction = prevention.get(global_pollutant, 0.0)
            regional_emissions = self.config.annual_emissions[global_pollutant][region]
            global_prevention_fraction += regional_emissions * regional_fraction

        if global_emissions > 0:
            global_prevention_fraction /= global_emissions

        global_emissions_after_prevention = global_emissions * (
            1.0 - min(global_prevention_fraction, 1.0)
        )

        natural_removal = {}
        incoming_transport = {}
        outgoing_transport = {}
        cleanup_removed = {}
        next_stocks = {}
        next_immobilized_stocks = {}
        mobile_stocks = {}
        immobilized_this_step = {}

        for pollutant in self.config.local_pollutants:
            old_stocks = self.stocks[pollutant]
            old_immobilized = self.immobilized_stocks[pollutant]
            natural_fraction = self.config.natural_removal_fraction[pollutant]
            after_natural_removal = []
            total_after_natural_removal = []
            natural_removal[pollutant] = []

            for region, stock in enumerate(old_stocks):
                emitted = emissions_after_prevention[pollutant][region]
                mobile_stock = stock - old_immobilized[region]
                mobile_stock += emitted * self.config.emission_mobile_fraction[pollutant]
                total_stock = stock + emitted
                removed = mobile_stock * natural_fraction
                natural_removal[pollutant].append(removed)
                total_after_natural_removal.append(total_stock - removed)
                after_natural_removal.append(mobile_stock - removed)

            matrix = self.config.transport_matrix[pollutant]
            outgoing = []
            incoming = [0.0] * len(self.config.regions)

            for source_region, source_stock in enumerate(after_natural_removal):
                total_outgoing = source_stock * sum(matrix[source_region])
                outgoing.append(total_outgoing)

                for destination_region, fraction in enumerate(matrix[source_region]):
                    incoming[destination_region] += source_stock * fraction

            outgoing_transport[pollutant] = tuple(outgoing)
            incoming_transport[pollutant] = tuple(incoming)
            cleanup_removed[pollutant] = []
            pollutant_next_stocks = []
            pollutant_next_immobilized = []
            pollutant_mobile_stocks = []
            immobilized_this_step[pollutant] = []

            for region in self.config.regions:
                local_after_transport = after_natural_removal[region]
                local_after_transport -= outgoing[region]
                local_after_transport += incoming[region]
                total_after_transport = total_after_natural_removal[region]
                total_after_transport -= outgoing[region]
                total_after_transport += incoming[region]
                cleanup_capacity = interventions[region]["cleanup_capacity"].get(
                    pollutant, 0.0
                )

                removed_by_cleanup = min(local_after_transport, cleanup_capacity)
                cleanup_removed[pollutant].append(removed_by_cleanup)
                mobile_after_cleanup = local_after_transport - removed_by_cleanup
                total_after_cleanup = total_after_transport - removed_by_cleanup
                immobilization_fraction = interventions[region][
                    "immobilization_fraction"
                ].get(pollutant, 0.0)
                newly_immobilized = mobile_after_cleanup * immobilization_fraction
                pollutant_next_immobilized.append(
                    old_immobilized[region] + newly_immobilized
                )
                immobilized_this_step[pollutant].append(newly_immobilized)
                pollutant_mobile_stocks.append(mobile_after_cleanup - newly_immobilized)
                pollutant_next_stocks.append(total_after_cleanup)

            next_stocks[pollutant] = tuple(pollutant_next_stocks)
            next_immobilized_stocks[pollutant] = tuple(pollutant_next_immobilized)
            mobile_stocks[pollutant] = tuple(pollutant_mobile_stocks)

        old_global_stock = self.stocks[global_pollutant]
        next_global_cohorts = []

        for age_years, mass in self.global_emission_cohorts:
            next_global_cohorts.append((age_years + 1, mass))

        next_global_cohorts.append((0, global_emissions_after_prevention))
        global_stock_before_removal_action = self._global_stock_from_cohorts(
            next_global_cohorts
        )
        global_natural_removal = (
            old_global_stock
            + global_emissions_after_prevention
            - global_stock_before_removal_action
        )
        global_cleanup_capacity = 0.0

        for intervention in interventions:
            global_cleanup_capacity += intervention["global_removal"]

        global_cleanup_removed = min(
            max(0.0, global_stock_before_removal_action),
            global_cleanup_capacity,
        )

        if global_cleanup_removed > 0:
            next_global_cohorts.append((0, -global_cleanup_removed))

        self.global_emission_cohorts = next_global_cohorts
        next_stocks[global_pollutant] = self._global_stock_from_cohorts(
            next_global_cohorts
        )
        natural_removal[global_pollutant] = global_natural_removal
        cleanup_removed[global_pollutant] = global_cleanup_removed
        incoming_transport[global_pollutant] = 0.0
        outgoing_transport[global_pollutant] = 0.0

        self.stocks = next_stocks
        self.immobilized_stocks = next_immobilized_stocks
        self.step_count += 1
        failed_pollutants = []

        for pollutant in self.config.local_pollutants:
            threshold = self.config.failure_thresholds[pollutant]

            for region, stock in enumerate(mobile_stocks[pollutant]):
                if stock >= threshold:
                    failed_pollutants.append((region, pollutant))

        if self.stocks[global_pollutant] >= self.config.failure_thresholds[global_pollutant]:
            failed_pollutants.append(("global", global_pollutant))

        terminated = len(failed_pollutants) > 0
        truncated = self.step_count >= self.config.episode_steps and not terminated
        self.done = terminated or truncated

        info = {
            "step": self.step_count,
            "time_step_name": self.config.time_step_name,
            "gross_emissions": self.config.annual_emissions,
            "emissions_after_prevention": emissions_after_prevention,
            "global_emissions_after_prevention": global_emissions_after_prevention,
            "natural_removal": natural_removal,
            "incoming_transport": incoming_transport,
            "outgoing_transport": outgoing_transport,
            "cleanup_removed": cleanup_removed,
            "stocks": self.stocks,
            "mobile_stocks": mobile_stocks,
            "immobilized_stocks": self.immobilized_stocks,
            "immobilized_this_step": immobilized_this_step,
            "failed_pollutants": tuple(failed_pollutants),
        }

        return self._observations(), terminated, truncated, info
