# Regional Pollution MARL

This project uses small synthetic simulations to explore how independent agents make decisions when one region's actions affect others. The model values are not calibrated to real economic or environmental data, so the results are not forecasts or policy recommendations.

## Study 1: Regional Emissions Coordination

[Study 1](study_1_emissions_coordination/README.md) compares Q-learning agents that choose low, medium, or high mitigation in a four-region network. It tests local, global, and mixed rewards, spillover, simple policies, and rewards for reducing harm to neighbors. The results describe the tested simulation settings and seeds.

## Study 2: Economic Development and Environmental Sustainability

[Study 2](study_2_economic_sustainability/README.md) adds regional currency and development, plus separate air, water, soil, and noise stocks. The experiments compare economic development with pollution prevention and cleanup. The candidate reward and refined EDI observations let the trained policies reach EDI 100 in most standard-emissions evaluations; a 20% emissions increase caused substantial soil failures, with the global-reward policies surviving most often.

Study 2 includes equation and fixed-policy feasibility screens, a decentralized baseline, reward tests, and evaluations of saved Q-learning policies. See its [README](study_2_economic_sustainability/README.md) for the results by experiment and for the distinctions between saved model versions.

## Project files

- [Methodology and tools](METHODOLOGY_AND_TOOLS.md): model equations, assumptions, experiment design, and findings for both studies.
- `study_1_emissions_coordination/`: Study 1 code, notebooks, and results.
- `study_2_economic_sustainability/`: Study 2 source code, notebooks, Q-tables, and results.

The units and coefficients are stylized assumptions. Interpret each result with the settings saved alongside its data, and only within that experiment's model version.
