# Study 2: Economic Development and Environmental Sustainability

## Research question

How do four independent regions balance development with pollution prevention and cleanup when pollution persists and moves between neighbors? This is a synthetic model, not a forecast of real economic or environmental outcomes.

## Current environment

The source environment has four regions arranged in a square. Each region has currency, an Economic Development Index (EDI), and separate air, water, soil, and noise stocks. At each step, all regions choose from 14 actions: maintain or develop, combined with no investment, low/medium/high prevention, or low/medium/high cleanup.

Prevention reduces new emissions; it does not remove existing pollution. Cleanup acts on existing stock and has a pollutant-specific efficiency. Development and mitigation costs are paid from the region's balance before that step's income is credited. Income is `20 + 0.5 * EDI`, development costs 100, and development gains are 10, 5, or 2 EDI below 40, 70, or 100 respectively. An episode ends after 200 steps or when any pollutant in any region reaches its synthetic limit of 100.

| Pollutant | Baseline emissions | Added per EDI | Transport share | Natural recovery | GPI weight |
|---|---:|---:|---:|---:|---:|
| Air | 3.0 | 0.08 | 30% | 60% | 35% |
| Water | 1.0 | 0.04 | 40% | 12% | 30% |
| Soil | 0.5 | 0.02 | 10% | 2% | 25% |
| Noise | 2.0 | 0.03 | 20% | 90% | 10% |

Each pollutant has a reference limit of 100. Prevention removes 0%, 25%, 50%, or 75% of new emissions at costs of 0, 10, 25, or 50. Cleanup costs 0, 15, 40, or 80 and has nominal capacity 0, 2, 5, or 10 units per pollutant. Transport redistributes remaining emissions and recovered stock between neighboring regions; it does not duplicate the transported amount. Full equations and assumptions are in the [methodology](../METHODOLOGY_AND_TOOLS.md#study-2-economic-development-and-multiple-pollutants).

The source `EnvironmentConfig` uses EDI observation boundaries `(34, 67)`. The trained policies in Notebook 18 use the refined boundaries `(34, 67, 70, 80, 90, 100)`. Use the matching configuration when loading their Q-tables. The other observation components are the region's currency bin, four local pollutant bins, and shared GPI bin. Agents have separate Q-tables and choose actions independently.

## Reward modes

The environment defaults to Notebook 17's candidate reward (`reward_formula="candidate"`, pollution-damage weight λ = 2.0). It rewards development gain, attained EDI, and income growth, and penalizes pollution damage, high pollutant pressure, paid development and mitigation costs, regional failure, and invalid actions. The earlier formula remains available with `reward_formula="original"`. Local, global, and mixed describe which score is used for Q-learning: local uses each region's score, global shares the mean score, and mixed combines 70% local with 30% global. The formulas are detailed in the [methodology](../METHODOLOGY_AND_TOOLS.md#rewards-and-learning).

## Experiments and main findings

| Notebook | What it tests | Main takeaway |
|---|---|---|
| 13 | Learned policies versus a decentralized threshold rule | The learned policies had lower mean GPI and higher final EDI than this one heuristic, with higher mitigation spending. This comparison does not isolate coordination as the cause. |
| 14 | Alternative emissions, income, development, and mitigation equations | The pooled target-EDI-60 success rate was 49.1% at multiplier 3.12, but policy rates ranged from 0% to 100%. The pooled rate is not a policy-robust balance point. |
| 15–16 | High-EDI policy tests and EDI-100 feasibility | The source equations can reach and maintain EDI 100 under tested fixed policies. Outcomes at high emissions also depend on the policy and starting cash. |
| 17 | Spillover and reward formulations | λ = 2.0 performed well under a fixed, non-learning lookahead policy. This does not establish how trained agents respond to the reward. |
| 18 | Refined EDI observations | With the reward and environmental equations unchanged, finer EDI bins removed the EDI-70 plateau in the trained policies. |
| 19 | Reward-mode behavior and a 20% emissions stress test | Global reward had the lowest GPI and lowest failure rate under stress; mixed reward most often reached EDI 100 at standard emissions. |

Notebook 18's standard-emissions evaluation used 100 episodes per reward mode (five training seeds and 20 evaluation seeds). The table reports means over those runs; all episodes survived.

| Reward mode | Final mean EDI | Mean GPI | Mitigation spending | All regions at EDI 100 |
|---|---:|---:|---:|---:|
| Local | 99.71 | 14.85 | 18,426 | 97% |
| Global | 98.61 | 9.02 | 28,065 | 72% |
| Mixed | 100.00 | 12.23 | 19,904 | 99% |

Global reward kept pollution lower on average but used the most mitigation and reached all-region EDI 100 less often. Mixed reward reached EDI 100 most consistently. These are differences among the saved learned policies, not a proof that reward mode alone caused every outcome.

Notebook 19 replayed these same policies without retraining and increased both baseline and EDI-linked emissions by 20%. Failure rates were 71% for local, 37% for global, and 73% for mixed reward; every failure involved soil. Mean final EDI was 85.59, 92.77, and 86.48, respectively. Global reward was more resilient in this stress test, alongside higher mitigation spending per episode. Failed episodes ended early, so total spending should be interpreted together with episode length.

Notebook 17's reward validation used a fixed, non-learning one-step lookahead controller. Both λ = 1.5 and λ = 2.0 reached EDI 100 and survived all 20 validation seeds. λ = 2.0 had slightly lower mean GPI (16.949 versus 17.196) and mean peak pressure (0.293 versus 0.295), with about 44 more currency spent on mitigation per episode. This supports λ = 2.0 as a candidate for further agent evaluation, not as evidence that learned agents will behave identically.

## Results and model versions

- `results/reward_candidate/` contains Notebook 03's candidate-reward run with the original coarse EDI bins. It shows the EDI-70 plateau.
- `results/edi_resolution/` contains Notebook 18's refined-observation Q-tables, settings, and matched evaluations, plus Notebook 19's action, trajectory, and emissions-stress summaries.
- `results/evaluation_episodes.csv` is an earlier run whose settings file does not record the reward formula. Do not use it as the candidate-reward result without independently resolving that version.
- `results/reward_comparison_current/` contains a historical single-pollutant experiment and is not comparable to the current four-stock model.

The equation and fixed-policy notebooks test candidate dynamics without changing the main environment source. Notebook 18 trained policies for its refined-observation condition; Notebook 19 only replayed saved policies and did not retrain them.

## Limits

The model uses invented costs, rates, weights, thresholds, and broad pollutant categories. A step is provisionally labeled as a year, but that time scale is not calibrated. Noise is treated as a persistent transported stock for convenience. The simulations omit pollutant species, weather, directional transport, watersheds, chemical transformations, and measured economic activity. Fixed-policy screens, one-step reward controllers, and five-seed learned-policy comparisons each answer narrower questions; none identifies a universally optimal policy.

## Code and outputs

- `config.py`, `environment.py`: source parameters and environment dynamics.
- `agents.py`, `train.py`, `evaluation.py`: tabular Q-learning, saving/loading, and evaluation.
- `formula_lab.py`, `policy_reward_comparison.py`: reusable equation and reward-comparison helpers.
- `notebooks/`: experiment setup, tables, plots, and interpretation.
- `results/`: settings, Q-tables, CSV summaries, and plots. Check the model version before reusing an artifact.
