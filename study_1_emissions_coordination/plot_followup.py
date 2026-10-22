"""Draw small SVG figures from the saved follow-up CSV files."""

import csv
from collections import defaultdict
from html import escape


OUTPUT_DIRECTORY = "study_1_emissions_coordination/results/followup"
FIGURE_DIRECTORY = "study_1_emissions_coordination/results/followup/figures"
POLICIES = ("fixed_medium", "random", "threshold", "iql_local", "iql_global", "iql_mixed")
COLORS = {"fixed_medium": "#1769aa", "random": "#626b75", "threshold": "#8751a3",
          "iql_local": "#d35400", "iql_global": "#16845b", "iql_mixed": "#bc2f45"}
ACTION_COLORS = ("#a6c8e0", "#4691bd", "#185780")


def read_csv(filename):
    with open(filename, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def start_svg(file, width, height, title):
    file.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
               f'viewBox="0 0 {width} {height}">\n')
    file.write('<rect width="100%" height="100%" fill="white"/>\n')
    file.write('<style>text{font:13px sans-serif;fill:#253341}.title{font:bold 19px sans-serif}'
               '.axis{stroke:#253341;stroke-width:1.5}</style>\n')
    file.write(f'<text class="title" x="55" y="35">{escape(title)}</text>\n')


def plot_actions():
    rows = read_csv(f"{OUTPUT_DIRECTORY}/diagnostics/action_counts_by_state.csv")
    counts = defaultdict(int)
    for row in rows:
        counts[(row["policy"], int(row["region"]), int(row["action"]))] += int(row["count"])

    with open(f"{FIGURE_DIRECTORY}/actions_by_region.svg", "w", encoding="utf-8") as file:
        start_svg(file, 930, 555, "Action choices by policy and region")
        file.write('<text x="55" y="61">Fraction of held-out region-steps; learned policies average five training seeds</text>\n')
        for policy_index, policy in enumerate(POLICIES):
            y = 95 + 72 * policy_index
            file.write(f'<text x="55" y="{y+20}">{escape(policy)}</text>\n')
            for region in range(4):
                x = 190 + 175 * region
                total = sum(counts[(policy, region, action)] for action in range(3))
                offset = 0.0
                for action in range(3):
                    fraction = counts[(policy, region, action)] / total
                    width = 125 * fraction
                    file.write(f'<rect x="{x+offset:.1f}" y="{y}" width="{width:.1f}" '
                               f'height="27" fill="{ACTION_COLORS[action]}"/>\n')
                    offset += width
                file.write(f'<text x="{x+42}" y="{y+47}">region {region}</text>\n')
        for action, color in enumerate(ACTION_COLORS):
            x = 190 + action * 145
            file.write(f'<rect x="{x}" y="525" width="17" height="17" fill="{color}"/>\n')
            file.write(f'<text x="{x+24}" y="538">action {action}</text>\n')
        file.write('</svg>\n')


def plot_trajectories():
    rows = read_csv(f"{OUTPUT_DIRECTORY}/diagnostics/matched_trajectory.csv")
    loads = defaultdict(list)
    for row in rows:
        loads[(row["policy"], int(row["step"]))].append(float(row["pollution_load"]))
    means = {key: sum(values) / len(values) for key, values in loads.items()}
    low = min(means.values()) - 2
    high = max(means.values()) + 2

    with open(f"{FIGURE_DIRECTORY}/matched_trajectories.svg", "w", encoding="utf-8") as file:
        start_svg(file, 940, 565, "Matched demand: one illustrative episode")
        file.write('<text x="55" y="62">Held-out demand seed 100000; learned and random policies use training/action seed 11</text>\n')
        file.write('<path class="axis" d="M70 90 V475 H725" fill="none"/>\n')
        for value in range(int(low), int(high) + 1, 5):
            y = 475 - (value - low) / (high - low) * 385
            file.write(f'<text x="59" y="{y+4:.1f}" text-anchor="end">{value}</text>\n')
        for step in (1, 10, 20, 30, 40, 50):
            x = 70 + (step - 1) / 49 * 655
            file.write(f'<text x="{x:.1f}" y="497" text-anchor="middle">{step}</text>\n')
        file.write('<text x="305" y="530">Episode step</text>\n')
        file.write('<text transform="translate(19 385) rotate(-90)">Mean pollution load</text>\n')
        for index, policy in enumerate(POLICIES):
            points = []
            for step in range(1, 51):
                x = 70 + (step - 1) / 49 * 655
                y = 475 - (means[(policy, step)] - low) / (high - low) * 385
                points.append(f"{x:.1f},{y:.1f}")
            file.write(f'<polyline points="{" ".join(points)}" fill="none" '
                       f'stroke="{COLORS[policy]}" stroke-width="2"/>\n')
            y = 100 + index * 29
            file.write(f'<path d="M750 {y} H775" stroke="{COLORS[policy]}" stroke-width="3"/>\n')
            file.write(f'<text x="783" y="{y+4}">{escape(policy)}</text>\n')
        file.write('</svg>\n')


def plot_uncertainty():
    for scenario, label in (("spillover_02", "0.2"), ("spillover_00", "0.0")):
        rows = read_csv(f"{OUTPUT_DIRECTORY}/{scenario}/policy_summary.csv")
        with open(f"{FIGURE_DIRECTORY}/{scenario}_pollution_vs_cost.svg", "w", encoding="utf-8") as file:
            start_svg(file, 930, 570, f"Pollution versus cost, spillover {label}")
            file.write('<text x="55" y="62">Points are means; bars show one SD across training runs</text>\n')
            file.write('<path class="axis" d="M80 90 V475 H700" fill="none"/>\n')
            costs = [float(row["mitigation_cost_mean"]) for row in rows]
            loads = [float(row["mean_regional_pollution_mean"]) for row in rows]
            cost_sds = [float(row["mitigation_cost_sd"]) for row in rows]
            load_sds = [float(row["mean_regional_pollution_sd"]) for row in rows]
            min_cost = min(value - sd for value, sd in zip(costs, cost_sds)) - 40
            max_cost = max(value + sd for value, sd in zip(costs, cost_sds)) + 40
            min_load = min(value - sd for value, sd in zip(loads, load_sds)) - 1
            max_load = max(value + sd for value, sd in zip(loads, load_sds)) + 1

            def x_position(value):
                return 80 + (value - min_cost) / (max_cost - min_cost) * 620

            def y_position(value):
                return 475 - (value - min_load) / (max_load - min_load) * 385

            for tick in range(5):
                cost = min_cost + tick * (max_cost - min_cost) / 4
                load = min_load + tick * (max_load - min_load) / 4
                file.write(f'<text x="{x_position(cost):.1f}" y="497" text-anchor="middle">{cost:.0f}</text>\n')
                file.write(f'<text x="70" y="{y_position(load)+4:.1f}" text-anchor="end">{load:.1f}</text>\n')
            file.write('<text x="280" y="535">Mitigation cost per episode</text>\n')
            file.write('<text transform="translate(19 370) rotate(-90)">Mean regional pollution</text>\n')
            for row in rows:
                policy = row["policy"]
                cost = float(row["mitigation_cost_mean"])
                load = float(row["mean_regional_pollution_mean"])
                cost_sd = float(row["mitigation_cost_sd"])
                load_sd = float(row["mean_regional_pollution_sd"])
                x, y = x_position(cost), y_position(load)
                file.write(f'<path d="M{x_position(cost-cost_sd):.1f} {y:.1f} '
                           f'H{x_position(cost+cost_sd):.1f} M{x:.1f} {y_position(load-load_sd):.1f} '
                           f'V{y_position(load+load_sd):.1f}" stroke="{COLORS[policy]}" '
                           f'stroke-width="2"/>\n')
                file.write(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{COLORS[policy]}"/>\n')
            for index, policy in enumerate(POLICIES):
                y = 105 + index * 30
                file.write(f'<circle cx="750" cy="{y}" r="5" fill="{COLORS[policy]}"/>\n')
                file.write(f'<text x="764" y="{y+4}">{escape(policy)}</text>\n')
            file.write('</svg>\n')


def plot_ablation():
    rows = read_csv(f"{OUTPUT_DIRECTORY}/ablation_summary.csv")
    rows = [row for row in rows if row["metric"] in
            ("mean_regional_pollution", "mitigation_cost")]
    with open(f"{FIGURE_DIRECTORY}/spillover_ablation.svg", "w", encoding="utf-8") as file:
        start_svg(file, 950, 460, "Global minus local: spillover comparison")
        file.write('<text x="55" y="61">Paired means and one SD across 20 training seeds; positive means global is higher</text>\n')
        for panel, metric in enumerate(("mean_regional_pollution", "mitigation_cost")):
            panel_rows = [row for row in rows if row["metric"] == metric]
            left = 55 + panel * 460
            low = min(float(row["mean"]) - float(row["sd"]) for row in panel_rows)
            high = max(float(row["mean"]) + float(row["sd"]) for row in panel_rows)
            padding = max((high - low) * 0.15, 0.1)
            low -= padding
            high += padding

            def x_position(value):
                return left + 130 + (value - low) / (high - low) * 275

            title = "Mean pollution difference" if panel == 0 else "Mitigation cost difference"
            file.write(f'<text x="{left}" y="110">{title}</text>\n')
            if low <= 0 <= high:
                x = x_position(0)
                file.write(f'<path d="M{x:.1f} 140 V350" stroke="#aaa" stroke-dasharray="4 4"/>\n')
            for index, row in enumerate(panel_rows):
                y = 170 + index * 85
                mean = float(row["mean"])
                sd = float(row["sd"])
                label = row["comparison"].replace("spillover_02_minus_00", "change in gap")
                label = label.replace("spillover_02", "spillover 0.2").replace("spillover_00", "spillover 0.0")
                file.write(f'<text x="{left}" y="{y+5}">{escape(label)}</text>\n')
                file.write(f'<path d="M{x_position(mean-sd):.1f} {y} '
                           f'H{x_position(mean+sd):.1f}" stroke="#1769aa" stroke-width="3"/>\n')
                file.write(f'<circle cx="{x_position(mean):.1f}" cy="{y}" r="6" fill="#1769aa"/>\n')
                file.write(f'<text x="{x_position(mean):.1f}" y="{y+26}" text-anchor="middle">{mean:.2f}</text>\n')
        file.write('</svg>\n')


def main():
    plot_actions()
    plot_trajectories()
    plot_uncertainty()
    plot_ablation()
    print(f"Saved figures to {FIGURE_DIRECTORY}")


if __name__ == "__main__":
    main()
