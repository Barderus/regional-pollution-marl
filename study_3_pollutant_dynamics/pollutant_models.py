"""Pollutant-specific annual equations used by the Study 3 environment."""
import math

CO2_RESPONSE_COMPONENTS = ((0.2173, None), (0.2240, 394.4), (0.2824, 36.54), (0.2763, 4.304))

def pm25_exposure(baseline_exposure_ug_m3, emission_change_tons, transfer_ug_m3_per_1000_tons, cross_region_share):
    exposure = []
    region_count = len(baseline_exposure_ug_m3)

    for receptor_region in range(region_count):
        local_change = emission_change_tons[receptor_region]
        other_region_change = 0.0

        for source_region in range(region_count):
            if source_region != receptor_region:
                other_region_change += emission_change_tons[source_region]

        local_contribution = local_change / 1000
        cross_contribution = other_region_change / 1000 * cross_region_share / (region_count - 1)
        concentration_change = transfer_ug_m3_per_1000_tons * (local_contribution + cross_contribution)
        receptor_exposure = baseline_exposure_ug_m3[receptor_region] + concentration_change

        if receptor_exposure < 0:
            receptor_exposure = 0.0

        exposure.append(receptor_exposure)

    return tuple(exposure)


def co2_remaining_fraction(age_years):
    remaining_fraction = 0.0

    for coefficient, time_constant in CO2_RESPONSE_COMPONENTS:
        if time_constant is None:
            remaining_fraction += coefficient
        else:
            remaining_fraction += coefficient * math.exp(-age_years / time_constant)

    return remaining_fraction


def co2_concentration(initial_concentration_ppm, background_growth_ppm_year, year_number, emission_cohorts_gtco2, ppm_per_gtco2):
    concentration = initial_concentration_ppm + background_growth_ppm_year * year_number

    for age_years, emission_deviation_gtco2 in emission_cohorts_gtco2:
        concentration += (emission_deviation_gtco2 * ppm_per_gtco2 * co2_remaining_fraction(age_years))

    return concentration


def nitrate_annual_step(stock_kg_n, annual_discharge_m3, water_volume_m3, annual_input_kg_n, denitrification_rate_per_year):
    discharge_rate_per_year = annual_discharge_m3 / water_volume_m3
    total_loss_rate_per_year = discharge_rate_per_year + denitrification_rate_per_year

    if total_loss_rate_per_year == 0:
        next_stock_kg_n = stock_kg_n + annual_input_kg_n
        outflow_kg_n = 0.0
        denitrified_kg_n = 0.0
    else:
        decay_fraction = math.exp(-total_loss_rate_per_year)
        next_stock_kg_n = (stock_kg_n * decay_fraction + annual_input_kg_n / total_loss_rate_per_year * (1 - decay_fraction))
        integrated_stock_kg_n_years = (stock_kg_n * (1 - decay_fraction) / total_loss_rate_per_year
                                       + annual_input_kg_n / total_loss_rate_per_year
                                       * (1 - (1 - decay_fraction) / total_loss_rate_per_year))

        outflow_kg_n = discharge_rate_per_year * integrated_stock_kg_n_years
        denitrified_kg_n = denitrification_rate_per_year * integrated_stock_kg_n_years

    next_concentration_mg_n_l = 1000 * next_stock_kg_n / water_volume_m3
    mass_balance_residual_kg_n = (stock_kg_n + annual_input_kg_n - next_stock_kg_n - outflow_kg_n - denitrified_kg_n)

    return {
        "stock_kg_n": next_stock_kg_n,
        "concentration_mg_n_l": next_concentration_mg_n_l,
        "outflow_kg_n": outflow_kg_n,
        "denitrified_kg_n": denitrified_kg_n,
        "mass_balance_residual_kg_n": mass_balance_residual_kg_n,
    }


def lead_annual_step(total_stock_kg_pb, annual_input_kg_pb, mobile_fraction, annual_mobile_export_fraction):
    stock_after_emission = total_stock_kg_pb + annual_input_kg_pb
    mobile_before_export_kg_pb = stock_after_emission * mobile_fraction
    exported_kg_pb = mobile_before_export_kg_pb * annual_mobile_export_fraction
    next_total_stock_kg_pb = stock_after_emission - exported_kg_pb
    next_mobile_stock_kg_pb = next_total_stock_kg_pb * mobile_fraction
    mass_balance_residual_kg_pb = (total_stock_kg_pb + annual_input_kg_pb - exported_kg_pb - next_total_stock_kg_pb)

    return {
        "total_stock_kg_pb": next_total_stock_kg_pb,
        "mobile_stock_kg_pb": next_mobile_stock_kg_pb,
        "exported_kg_pb": exported_kg_pb,
        "mass_balance_residual_kg_pb": mass_balance_residual_kg_pb,
    }
