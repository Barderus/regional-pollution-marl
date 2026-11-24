"""Configuration schema for the Study 3 pollutant dynamics stage."""


class EnvironmentConfig:
    def __init__(self, parameters):
        required_sections = (
            "regions",
            "local_pollutants",
            "global_pollutant",
            "initial_stocks",
            "initial_immobilized_stocks",
            "annual_emissions",
            "emission_mobile_fraction",
            "local_natural_removal_fraction",
            "transport_matrix",
            "failure_thresholds",
            "interventions",
            "episode_steps",
        )

        missing_sections = []

        for section in required_sections:
            if section not in parameters:
                missing_sections.append(section)

        if missing_sections:
            missing_text = ", ".join(missing_sections)
            raise ValueError(f"Missing configuration sections: {missing_text}")

        self.regions = tuple(parameters["regions"])
        self.local_pollutants = tuple(parameters["local_pollutants"])
        self.global_pollutant = parameters["global_pollutant"]
        self.pollutants = self.local_pollutants + (self.global_pollutant,)
        self.initial_stocks = parameters["initial_stocks"]
        self.initial_immobilized_stocks = parameters["initial_immobilized_stocks"]
        self.annual_emissions = parameters["annual_emissions"]
        self.emission_mobile_fraction = parameters["emission_mobile_fraction"]
        self.natural_removal_fraction = parameters["local_natural_removal_fraction"]
        self.transport_matrix = parameters["transport_matrix"]
        self.failure_thresholds = parameters["failure_thresholds"]
        self.interventions = parameters["interventions"]
        self.episode_steps = parameters["episode_steps"]
        self.time_step_name = "year"

        self._check_parameters()

    def _check_parameters(self):
        if len(self.regions) < 2:
            raise ValueError("Study 3 requires at least two regions")

        if self.regions != tuple(range(len(self.regions))):
            raise ValueError("Regions must be indexed from zero in the configured order")

        if self.global_pollutant in self.local_pollutants:
            raise ValueError("The global pollutant cannot also be local")

        if self.episode_steps <= 0:
            raise ValueError("episode_steps must be positive")

        for pollutant in self.pollutants:
            for section_name, section in (
                ("initial_stocks", self.initial_stocks),
                ("annual_emissions", self.annual_emissions),
                ("failure_thresholds", self.failure_thresholds),
            ):
                if pollutant not in section:
                    raise ValueError(f"{section_name} is missing {pollutant}")

            if self.failure_thresholds[pollutant] <= 0:
                raise ValueError("Failure thresholds must be positive")

            initial_values = self.initial_stocks[pollutant]
            emission_values = self.annual_emissions[pollutant]

            if pollutant == self.global_pollutant:
                if initial_values < 0:
                    raise ValueError("Initial global stock cannot be negative")

                if len(emission_values) != len(self.regions):
                    raise ValueError("Global emissions must have one value per region")
            else:
                if pollutant not in self.natural_removal_fraction:
                    raise ValueError(
                        f"local_natural_removal_fraction is missing {pollutant}"
                    )

                if not 0 <= self.natural_removal_fraction[pollutant] <= 1:
                    raise ValueError("Local annual removal fractions must be 0 to 1")

                if pollutant not in self.initial_immobilized_stocks:
                    raise ValueError(f"initial_immobilized_stocks is missing {pollutant}")

                if len(initial_values) != len(self.regions):
                    raise ValueError("Local initial stocks must match the region count")

                immobilized_values = self.initial_immobilized_stocks[pollutant]

                if len(immobilized_values) != len(self.regions):
                    raise ValueError("Initial immobilized stocks must match regions")

                if len(emission_values) != len(self.regions):
                    raise ValueError("Local emissions must match the region count")

                if any(value < 0 for value in initial_values):
                    raise ValueError("Initial stocks cannot be negative")

                for region, immobilized in enumerate(immobilized_values):
                    if immobilized < 0 or immobilized > initial_values[region]:
                        raise ValueError("Immobilized stock must be within total stock")

                if pollutant not in self.emission_mobile_fraction:
                    raise ValueError(f"emission_mobile_fraction is missing {pollutant}")

                if not 0 <= self.emission_mobile_fraction[pollutant] <= 1:
                    raise ValueError("Emission mobile fractions must be between 0 and 1")

            if any(value < 0 for value in emission_values):
                raise ValueError("Annual emissions cannot be negative")

        for pollutant in self.local_pollutants:
            if pollutant not in self.transport_matrix:
                raise ValueError(f"transport_matrix is missing {pollutant}")

            matrix = self.transport_matrix[pollutant]

            if len(matrix) != len(self.regions):
                raise ValueError("Transport matrix rows must match the region count")

            for source_region, row in enumerate(matrix):
                if len(row) != len(self.regions):
                    raise ValueError("Transport matrix columns must match regions")

                if row[source_region] != 0:
                    raise ValueError("Transport cannot move pollution to the same region")

                if any(fraction < 0 for fraction in row):
                    raise ValueError("Transport fractions cannot be negative")

                if sum(row) > 1:
                    raise ValueError("A region cannot transport more than its full stock")

        for intervention_name, intervention in self.interventions.items():
            required_intervention_values = (
                "prevention_fraction",
                "cleanup_capacity",
                "immobilization_fraction",
                "global_removal",
            )

            for value_name in required_intervention_values:
                if value_name not in intervention:
                    raise ValueError(
                        f"Intervention {intervention_name} is missing {value_name}"
                    )

            for pollutant, fraction in intervention["prevention_fraction"].items():
                if pollutant not in self.pollutants:
                    raise ValueError(f"Unknown pollutant in {intervention_name}")

                if fraction < 0 or fraction > 1:
                    raise ValueError("Prevention fractions must be between 0 and 1")

            for pollutant, capacity in intervention["cleanup_capacity"].items():
                if pollutant not in self.local_pollutants:
                    raise ValueError("Local cleanup must target a local pollutant")

                if capacity < 0:
                    raise ValueError("Cleanup capacity cannot be negative")

            for pollutant, fraction in intervention["immobilization_fraction"].items():
                if pollutant not in self.local_pollutants:
                    raise ValueError("Immobilization must target a local pollutant")

                if fraction < 0 or fraction > 1:
                    raise ValueError("Immobilization fractions must be between 0 and 1")

            if intervention["global_removal"] < 0:
                raise ValueError("Global removal capacity cannot be negative")
