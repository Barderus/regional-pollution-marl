"""Project timing and pollutant-specific investment effects for Study 3."""

from study_3_pollutant_dynamics.mitigation_parameters import (MITIGATION_PROJECTS, PREVENTION_FRACTION_CAP)


def project_is_active(project, year):
    parameters = MITIGATION_PROJECTS[project["category"]][project["pollutant"]]
    active_year = project["investment_year"] + parameters["delay_years"]
    final_active_year = active_year + parameters["persistence_years"]
    is_active = active_year <= year < final_active_year

    return is_active


def active_project_count(projects, year, category, pollutant):
    active_count = 0

    for project in projects:
        if project["category"] != category or project["pollutant"] != pollutant:
            continue

        if project_is_active(project, year):
            active_count += 1

    return active_count


def prevention_fraction(projects, year, pollutant):
    active_count = active_project_count(projects, year, "prevention", pollutant)
    effectiveness = MITIGATION_PROJECTS["prevention"][pollutant]["emissions_reduction_fraction"]
    fraction = min(PREVENTION_FRACTION_CAP, active_count * effectiveness)

    return fraction


def cleanup_capacity(projects, year, pollutant, reference_emissions, initial_stock):
    active_count = active_project_count(projects, year, "cleanup", pollutant)
    capacity = 0.0

    if active_count == 0:
        return capacity

    project_parameters = MITIGATION_PROJECTS["cleanup"][pollutant]

    if pollutant == "CO2":
        capacity = (reference_emissions * project_parameters["capacity_fraction_reference_emissions"] * active_count)
    else:
        capacity = (initial_stock * project_parameters["capacity_fraction_initial_stock"] * active_count)

    return capacity


def lead_mobile_fraction(projects, year, base_mobile_fraction):
    active_count = active_project_count(projects, year, "containment", "Lead")
    reduction_multiplier = MITIGATION_PROJECTS["containment"]["Lead"]["mobile_fraction_multiplier"]
    mobile_fraction = base_mobile_fraction * (reduction_multiplier ** active_count)

    return mobile_fraction


def purchase_costs(policy_projects):
    total_cost = 0.0

    for category, pollutant in policy_projects:
        total_cost += MITIGATION_PROJECTS[category][pollutant]["cost"]

    return total_cost
