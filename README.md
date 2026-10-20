# Regional Pollution Coordination with Multi-Agent RL

## Introduction

This project looks at how four neighboring regions coordinate pollution mitigation. Each region can reduce its own emissions, while pollution persists and spreads to its neighbors. The model is synthetic, uses arbitrary units, and does not represent a real climate system.

## Goal

The goal is to see how the regions' choices change under local, global, and mixed rewards. Each region chooses one of three mitigation levels using only its own demand category and pollution bin. Independent Q-learning agents are trained under each reward and compared with random, fixed-medium, and threshold policies.

The initial hypotheses were that shared rewards might reduce pollution and local rewards might keep mitigation costs lower. The results below test those expectations.

## Results

Every policy was evaluated on the same 20 held-out demand sequences. Each Q-learning condition used five training seeds, and the random policy used five action seeds. Fixed medium and the threshold rule are deterministic. The table shows average outcomes.

| Policy | Mean regional pollution | Mitigation cost | Fraction above 30 |
|---|---:|---:|---:|
| Fixed medium | 23.88 | 800.0 | 0.142 |
| Random | 23.94 | 929.3 | 0.152 |
| Q-learning, global reward | 25.08 | 718.0 | 0.186 |
| Q-learning, local reward | 25.82 | 562.2 | 0.217 |
| Threshold rule | 25.96 | 540.2 | 0.221 |
| Q-learning, mixed reward | 26.01 | 557.2 | 0.234 |

In this run, global-reward Q-learning produced lower pollution than local- or mixed-reward Q-learning, with higher mitigation cost. Fixed medium had the lowest pollution overall, at a higher cost than any learned policy. The [pollution-versus-cost plot](results/figures/pollution_vs_cost.svg) shows this tradeoff, and the [full summary](results/raw/policy_summary.csv) includes variation across seeds.

## To do

- [ ] Examine action choices by region and pollution trajectories over an episode.
- [ ] Show seed-to-seed variation on the pollution-versus-cost plot.
- [ ] Use more training seeds to check whether these patterns hold.
- [ ] Explore other coefficients and region connections in separate experiments.

## Metrics

- **Mean regional pollution:** the average pollution load after each step, across all regions.
- **Mitigation cost:** total action cost across all regions and steps in an episode.
- **Threshold exceedance:** the share of region-steps with pollution above 30.
- **Emitted units:** total emissions during an episode.
- **Regional pollution spread:** the gap between the highest and lowest regional average pollution.

Action frequencies are also saved. The [episode data](results/raw/episodes.csv) are averaged within each seed in the [run summaries](results/raw/run_summary.csv), then across seeds in the full policy summary. These common metrics allow comparison across reward setups with different training rewards.

## Limits

The study uses one square network, fixed coefficients, and agents with limited local observations. The learned-policy results come from five training seeds and 20 held-out demand sequences. They describe this synthetic setting and do not establish which reward design would work best elsewhere or predict real-world emissions.
