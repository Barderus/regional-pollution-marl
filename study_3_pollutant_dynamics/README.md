# Study 3: Pollutant Dynamics

## Research question and scope

Study 3 explores how economic development, pollutant dynamics, mitigation, and environmental damage interact across four U.S. Census regions: Northeast, Midwest, South, and West. It is a synthetic annual simulation, not a forecast of real regional environmental or economic outcomes. Census regions are broad groupings and are not modeled as actual airsheds, watersheds, or soil parcels.

The four pollutants are PM2.5, atmospheric CO2, nitrate-N in water, and lead in soil. Their physical units and environmental processes differ, so the implementation models each separately. A shared CO2 concentration is exposed to all four regions. See [source and parameter specification](source_and_parameter_specification.md) for the evidence review and parameter classifications.

## Current environment

The pollutant-only model in `environment.py` uses the equations implemented in `pollutant_models.py`. PM2.5 is calculated as annual exposure from regional emissions through a source-receptor approximation. CO2 is a shared atmospheric concentration path with regional emissions deviations represented using the Joos et al. multi-timescale impulse response. Nitrate-N is represented as mass in a well-mixed receiving-water compartment with flow and denitrification. Soil lead is tracked as persistent total mass with a mobile fraction and export. The model supports configurable scenario parameters; its simplifications and assumptions are documented in `scenario_parameters.py` and `source_and_parameter_specification.md`.

The integrated model in `integrated_environment.py` combines those pollutant processes with regional EDI, currency, income, environmental damage, and mitigation project purchases. Regions choose among eight actions, including maintaining, developing, prevention, and investment-package combinations. Development and project costs are charged against available currency; a retained share of post-damage income is added afterward. Environmental damage reduces current income and can carry forward into the income base. Development costs can be constant, linear, or quadratic. Damage rates, income assumptions, regional starting conditions, mitigation costs, and project effectiveness are scenario choices, not calibrated regional estimates.

| Pollutant | Modeled quantity | Current process representation |
|---|---|---|
| PM2.5 | Annual exposure (micrograms/m³) | Background exposure plus emissions mapped to receptor regions; recalculated annually rather than carried as a mass stock. |
| CO2 | Shared atmospheric concentration (ppm) | Observed starting concentration and background path, plus regional emissions deviations convolved with a published impulse-response function. |
| Nitrate-N | Water concentration (mg N/L) and mass (kg N) | A simplified water compartment with annual loading, discharge flushing, denitrification, and mitigation removal. |
| Lead | Total soil mass (kg Pb) and mobile concentration | Persistent soil mass with assumed mobile partition and export; cleanup removes mass and containment changes mobility. |

There are no environmental failure thresholds in the current integrated baseline. Episodes end at the configured horizon. The current environment does not include irreversible regional or global collapse, sector-specific development, transfers between regions, catastrophe events, or disaster preparedness.

## Experiments and current findings

| Notebook | What it examines | Status / interpretation |
|---|---|---|
| 00 to 05 | Study overview; pollutant specification, equation audits, parameter ranges, and integrated pollutant trajectories | Pollutant equations have been implemented and checked for expected mathematical behavior under hypothetical scenarios. Notebook 05 compares reference and increased-emissions paths over 40 years. These checks are not empirical validation. |
| 06 | Regional income and emissions activity assumptions | Tests provisional regional economic equations; regional values are synthetic and emissions indices are not physical loads. |
| 07, 11 to 13 | Pollution damage to income, income persistence and retention, and development cost curves | Sensitivity analyses of synthetic damage and economic assumptions. They do not estimate a causal regional damage function. |
| 08 to 10 | Mitigation projects, integrated fixed-policy baselines, and development cost functions | Tests deterministic policies and transition accounting. Project prices, capacities, effectiveness, and lifetimes remain assumptions. |
| 14 to 17 | Tabular Q-learning formulation, training coverage and duration, mixed-condition training, and observation variants | RL pilots and saved evaluations exist. Each region uses an independent tabular agent with local observations and local, global, or mixed rewards. These exploratory runs do not establish a selected reward or robust optimal policy. |

The integrated model is a reviewable synthetic baseline with pollutant, economic, damage, and mitigation components. Data sources provide some starting anchors, but they do not calibrate all regional parameters or model relationships. Findings from fixed policies and RL pilots are conditional on the chosen equations, assumptions, observations, and training settings. Consult notebook outputs and saved settings before reusing any result.

## Planned environmental redesign

`environmental changes.md` records the requested next-stage design work: evidence-informed severity functions and pollution indices, regional and global collapse mechanics, permanent damage and countdowns, environmental effects on development, sector-specific development, voluntary financial assistance, catastrophic events, and preparedness. It also specifies that the first deliverable is a complete design proposal in `study3_environment_design.md`, followed by review before implementation. None of these mechanisms should be read as part of the current environment described above.

The proposal must distinguish empirical observations, derived proxies, and synthetic design parameters; preserve pollutant-specific units; and assess aggregation, thresholds, and parameter uncertainty before implementation. The existing model and saved experiments are to be preserved during the planned staged work.

## Limitations

- Regional processes use stylized compartments and connectivity; they do not resolve actual airsheds, watersheds, soil conditions, or within-region variation.
- Scenario assumptions and derived proxies are not equivalent to calibrated model parameters. The annual time step and several spatial conversions are modeling choices.
- PM2.5 source-receptor coefficients, nitrate hydrology, lead soil parcel and mobility assumptions, and regional economic relationships remain limited or assumed.
- Damage functions and mitigation project attributes are synthetic sensitivity parameters. EDI is not GDP.
- Fixed-policy and RL outcomes depend on the implemented reward, state discretization, action set, and training conditions. They are exploratory model comparisons, not policy recommendations.

## Code, notebooks, and outputs

- `config.py`, `scenario_parameters.py`, and parameter modules define configuration and low/reference/high scenario inputs.
- `pollutant_models.py`, `environment.py`: pollutant dynamics and the pollutant-only environment.
- `regional_economics.py`, `environmental_damage.py`, `mitigation.py`, `integrated_environment.py`: economic processes, damage, projects, and integrated transitions.
- `agents.py`, `train.py`, `evaluation.py`, `rl_formulation.py`, `dqn_agent.py`, `dqn_training.py`, `ppo_agent.py`, `ppo_training.py`, `rl_experiments.py`: tabular Q-learning, DQN, PPO, experiment helpers, and evaluation. The interactive `run_training.py` entry point offers each as an independent-region training option. Adding PPO support makes it available for experiments; it does not select PPO for the environmental redesign.
- `notebooks/`: sequential audits, fixed-policy experiments, and RL pilot analyses (00 to 17).
- `data/`, `research_package/`: source datasets and research-package material; check source definitions and units before using values.
- `results/`: saved pilot outputs and model artifacts. Match each artifact to its settings and model version before interpreting or reusing it.\n