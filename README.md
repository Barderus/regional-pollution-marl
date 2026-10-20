# Regional Pollution Coordination with Multi-Agent RL

## Introduction

This project looks at how four neighboring regions coordinate pollution mitigation. Each region can reduce its own emissions, while pollution persists and spreads to its neighbors. The model is synthetic, uses arbitrary units, and does not represent a real climate system.

## Goal

The goal is to see how the regions' choices change under local, global, and mixed rewards. Each region chooses one of three mitigation levels using only its own demand category and pollution bin. Independent Q-learning agents are trained under each reward and compared with random, fixed-medium, and threshold policies.

The initial hypotheses were that shared rewards might reduce pollution and local rewards might keep mitigation costs lower. The results below test those expectations.

## Results

The expanded experiment uses 20 training seeds for each Q-learning condition and 20 action seeds for the random policy. Every run faces the same 20 held-out demand sequences. Fixed medium and the threshold rule are deterministic. With the original spillover value of 0.2, the results are:

| Policy | Mean regional pollution | Mitigation cost | Fraction above 30 |
|---|---:|---:|---:|
| Fixed medium | 23.88 | 800.0 | 0.142 |
| Random | 23.91 | 932.3 | 0.149 |
| Q-learning, global reward | 24.97 | 757.4 | 0.197 |
| Q-learning, mixed reward | 25.86 | 573.8 | 0.223 |
| Threshold rule | 25.96 | 540.2 | 0.221 |
| Q-learning, local reward | 26.18 | 527.5 | 0.252 |

Global-reward Q-learning produced lower pollution than local-reward Q-learning, with higher mitigation cost. Across paired training seeds, the global-minus-local pollution difference was −1.21 on average, with a standard deviation of 1.07; the cost difference was +230, with a standard deviation of 137. Global had lower pollution in 18 of 20 paired runs and higher cost in all 20. Fixed medium had the lowest average pollution in this sample. The [pollution-versus-cost plot](results/followup/figures/spillover_02_pollution_vs_cost.svg) shows the means and seed-to-seed spread, and the [full summary](results/followup/spillover_02/policy_summary.csv) includes all metrics.

The [action-choice figure](results/followup/figures/actions_by_region.svg) uses the original five saved training seeds. Global-reward agents chose the lowest mitigation action in 42% of region-steps, compared with 60% for local-reward agents; they chose the highest action in 17%, compared with 12%. That helps explain the pollution and cost tradeoff. The [matched-demand trajectory](results/followup/figures/matched_trajectories.svg) shows one illustrative episode. In that episode, global reward had slightly *more* pollution than local reward (25.58 versus 25.34), a reminder that one trajectory can differ from the aggregate. The [state-level action counts](results/followup/diagnostics/action_counts_by_state.csv) and [regional trajectory data](results/followup/diagnostics/matched_trajectory.csv) retain the detail behind the figures.

We also repeated training with neighbor spillover set to zero. Global-minus-local pollution was −1.23 there; global had lower pollution in 19 of 20 paired runs and higher cost in all 20. The change in the pollution gap between spillover settings was +0.03 with a standard deviation of 1.61 across paired seeds. The [ablation figure](results/followup/figures/spillover_ablation.svg) and [paired summary](results/followup/ablation_summary.csv) show that comparison. There is no clear change in the global-versus-local pollution gap in this experiment. Absolute pollution levels differ between settings, so the comparisons above are made within each setting.

## To do

- [x] Examine action choices by region and pollution trajectories over an episode.
- [x] Show seed-to-seed variation on the pollution-versus-cost plot.
- [x] Use more training seeds and compare the zero-spillover setting.
- [ ] Explore other coefficients and region connections in separate experiments.

## Metrics

- **Mean regional pollution:** the average pollution load after each step, across all regions.
- **Mitigation cost:** total action cost across all regions and steps in an episode.
- **Threshold exceedance:** the share of region-steps with pollution above 30.
- **Emitted units:** total emissions during an episode.
- **Regional pollution spread:** the gap between the highest and lowest regional average pollution.

Action frequencies are also saved. The [episode data](results/followup/spillover_02/episodes.csv) are averaged within each seed in the [run summaries](results/followup/spillover_02/run_summary.csv), then across seeds in the policy summary. The [zero-spillover results](results/followup/spillover_00/policy_summary.csv) use the same evaluation procedure. These common metrics allow comparison across reward setups with different training rewards.

## Limits

The study uses one square network, fixed coefficients apart from the spillover ablation, and agents with limited local observations. The expanded results come from 20 training seeds and 20 held-out demand sequences. The example trajectory is one episode, and seed-to-seed spread is substantial for some learned policies. These results describe this synthetic setting; they do not establish which reward design would work best elsewhere or predict real-world emissions.
