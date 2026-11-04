# Study 2: Economic Development and Environmental Sustainability

This study explores how independent agents balance regional development with
environmental investment when pollution can move between neighboring regions.
The environment is a synthetic decision-making model, not a prediction of real
environmental outcomes.

## Current model

The four regions track air, water, soil, and noise pollution separately. Each
generates baseline emissions every year, and economic development adds to those
emissions. Emissions can be prevented before release; cleanup acts on pollution
already present. Both emissions and recovered pollution stocks move between
neighbors through transfers deducted from the source region.

Each agent chooses whether to develop and whether to make one environmental
investment in the same step. The Global Pollution Index combines the four
pollutant loads after normalizing them against model-specific reference limits.
Individual pollutant stocks remain available for analysis.

One step currently represents one year as a working assumption. Emissions,
recovery rates, costs, weights, and reference limits are exploratory synthetic
values; they have not been calibrated to a specific place, pollutant species,
or real-world dataset. Previous notebook results use earlier environment
versions and do not describe this model.

## Training

`train.py` contains the Q-learning function and Q-table save/load helpers.
Notebook 03 handles evaluation and writes the settings and held-out episode
results. It stores the local, global, and mixed policies together in
`results/q_tables_notebook03.json` and writes one evaluation CSV.

The notebooks remain the main place to choose and inspect analyses. Their
training cells call the shared functions in `train.py` instead of maintaining
separate Q-learning loops.

## Current status

The multi-pollutant mechanics and shared training workflow are implemented.
The next step is to review the equations and parameter assumptions, then run
the model's fixed-policy validation before training the reward comparison.
Earlier results are historical and should not be used as evidence for the
current design.
