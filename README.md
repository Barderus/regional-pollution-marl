# Regional Pollution MARL

This repository explores how independent agents make decisions when their individual goals can affect the rest of the system. I'm using Multi-Agent Reinforcement Learning (MARL) to simulate neighboring regions that must balance their own interests with shared environmental challenges.

The project is divided into two studies. Both use a simulated environment with four neighboring regions, but explore different aspects of decentralized decision-making.

## Study 1: Regional Emissions Coordination

The [first study](study_1_emissions_coordination/README.md) focuses on how different reward structures influence pollution mitigation.

Each region is controlled by an independent Q-learning agent that decides how much to reduce its emissions. I'm comparing local, global, and mixed rewards to see how agents behave when prioritizing their own region versus the overall system.

The initial results show that global rewards generally lead to lower pollution than local rewards, but with higher mitigation costs. Interestingly, simpler strategies also performed well, raising questions about when reinforcement learning provides an advantage.

The study also explores how pollution spreading between regions affects agent behavior.

## Study 2: Economic Development and Environmental Sustainability

The [second study](study_2_economic_sustainability) builds on the first by introducing economic development and resource management.

Alongside pollution mitigation, agents decide when to invest in development, save currency, or pay to reduce emissions.

The challenge is that regions want to grow economically, but excessive pollution in any one region can cause the entire system to fail.

The initial comparison tests local, global, and mixed rewards against simple policies.

## Project Direction

Both studies are part of my broader interest in decentralized multi-agent decision-making, particularly how agents balance individual and shared objectives with limited information.

As the project develops, I want to explore different environments, regional connections, and coordination strategies to better understand when independent agents can achieve better outcomes through cooperation.

These are synthetic simulations designed to explore agent behavior, not predict real-world pollution or economic outcomes.
