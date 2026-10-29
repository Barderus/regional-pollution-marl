# Regional Pollution MARL

This project explores how independent agents make decisions when their actions affect other regions. I'm using Multi-Agent Reinforcement Learning (MARL) to study how agents balance their own interests with shared environmental challenges.

The project is divided into two studies. Both use a simulated environment with four neighboring regions, but focus on different questions about decentralized decision-making.

## Study 1: Regional Emissions Coordination

The [first study](study_1_emissions_coordination/README.md) focuses on how different reward structures influence pollution mitigation.

Each region has an independent Q-learning agent that decides how much to reduce its emissions. I compare local, global, and mixed rewards to see how agents behave when prioritizing their own region versus the overall system.

The initial results show that global rewards generally lead to lower pollution than local rewards, but with higher mitigation costs. Simpler strategies also performed well, raising questions about when reinforcement learning provides an advantage.

The study also looks at how pollution spreading between neighboring regions affects agent behavior.

## Study 2: Economic Development and Environmental Sustainability

The [second study](study_2_economic_sustainability/README.md) builds on the first by introducing economic development and resource management.

Each region must decide whether to invest in economic growth, spend currency on pollution mitigation, or save for future decisions.

The challenge is that development increases emissions, pollution spreads between regions, and excessive pollution in any region can cause the entire system to fail.

I've tested different environment settings, pollution mechanics, and economic rewards to create a trade-off between development and sustainability.

The next step is to compare local, global, and mixed rewards under the updated environment to see how agents balance economic growth with environmental constraints.

## Project Direction

Both studies are part of my interest in decentralized multi-agent decision-making, especially how agents balance individual and shared objectives.

As I continue working on this project, I want to explore how different reward structures, environmental conditions, and coordination strategies affect the decisions agents make.

The goal is to better understand when independent agents can learn to make decisions that benefit both themselves and the larger system.

These are synthetic simulations designed to study agent behavior, not predict real-world environmental or economic outcomes.