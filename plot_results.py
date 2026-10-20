# Draw a dependency-free SVG of held-out pollution versus mitigation cost

import csv
from html import escape


def plot_pollution_vs_cost(results_path, output_path):
    with open(results_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No policy summary rows")

    costs = [float(row["mitigation_cost_mean"]) for row in rows]
    loads = [float(row["mean_regional_pollution_mean"]) for row in rows]
    min_x, max_x = min(costs), max(costs)
    min_y, max_y = min(loads), max(loads)
    pad_x = max(10.0, (max_x - min_x) * 0.12)
    pad_y = max(1.0, (max_y - min_y) * 0.12)
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    # Map metric values into the SVG plot area (90..690 by 90..490).
    def cost_position(value):
        return 90 + (value - min_x) / (max_x - min_x) * 600

    def pollution_position(value):
        return 490 - (value - min_y) / (max_y - min_y) * 400

    colors = ["#1769aa", "#d35400", "#16845b", "#8751a3", "#bc2f45", "#626b75"]

    with open(output_path, "w", encoding="utf-8") as file:
        file.write(
            '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="580" viewBox="0 0 900 580">\n'
            '<rect width="900" height="580" fill="white"/>\n'
            '<style>text{font:14px sans-serif;fill:#27313d}.title{font:bold 20px sans-serif}.axis{stroke:#27313d;stroke-width:2}</style>\n'
            '<text class="title" x="90" y="40">Held-out pollution versus mitigation cost</text>\n'
            '<path class="axis" d="M90 90 V490 H690" fill="none"/>\n'
            '<text x="320" y="550">Mitigation cost per episode (arbitrary units)</text>\n'
            '<text transform="translate(25 370) rotate(-90)">Mean regional pollution (arbitrary units)</text>\n'
        )

        for i in range(5):
            cost_tick = min_x + i * (max_x - min_x) / 4
            pollution_tick = min_y + i * (max_y - min_y) / 4
            file.write(f'<text x="{cost_position(cost_tick):.1f}" y="515" text-anchor="middle">{cost_tick:.0f}</text>\n')
            file.write(f'<text x="78" y="{pollution_position(pollution_tick)+5:.1f}" text-anchor="end">{pollution_tick:.1f}</text>\n')

        for i, row in enumerate(rows):
            point_x = cost_position(costs[i])
            point_y = pollution_position(loads[i])
            label = escape(row["policy"])
            color = colors[i % len(colors)]
            file.write(f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="6" fill="{color}"/>\n')
            file.write(f'<text x="{point_x+10:.1f}" y="{point_y-9:.1f}">{label}</text>\n')

        file.write('</svg>\n')


def main():
    results_path = "results/raw/policy_summary.csv"
    output_path = "results/figures/pollution_vs_cost.svg"
    plot_pollution_vs_cost(results_path, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
