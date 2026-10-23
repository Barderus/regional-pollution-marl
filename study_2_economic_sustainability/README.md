# Study 2: Economic Development and Environmental Sustainability

**Status: Environment implemented; agents and experiments planned**

## Overview

This study builds on [Study 1](../study_1_emissions_coordination/README.md) by introducing economic development and resource management alongside pollution mitigation.

The idea is that each region wants to grow economically, but development also increases emissions. Regions need to decide whether to invest in their economy, spend resources on reducing pollution, or save their currency for future decisions.

The challenge is that pollution doesn't stay within regional boundaries, and if pollution in any region reaches a critical level, the entire system fails.

I'm interested in exploring how independent agents balance their own economic growth with the environmental needs of the larger system, especially when their decisions can have consequences for neighboring regions.

## The Environment

The simulation has four neighboring regions. Each will eventually be controlled by an independent Q-learning agent.

Each agent will choose between five actions:

- **No investment:** Save currency for future decisions.
- **Development:** Invest in economic growth, increasing future income but also emissions.
- **Low, medium, or high mitigation:** Spend currency to reduce pollution at different levels.

Agents will have access to their own economic development, available currency, and pollution levels, along with a shared Global Pollution Index (GPI).

As in Study 1, I'll compare three reward structures:

- **Local:** Agents prioritize their own regional outcomes.
- **Global:** Agents consider the overall system.
- **Mixed:** Agents balance local and global objectives.

Unlike Study 1, mitigation and development will require agents to spend currency they've earned, making resource management an important part of their decisions.

## Current status

The environment and its settings are in `environment.py` and `config.py`. It accepts four actions at a time and returns each region's next observation and reward, separate termination and time-limit flags, and the underlying quantities used in the transition. With a reset seed, the four starting currency balances can be reproduced. The mechanics still need focused validation before agent training.

## Next Steps

After checking the environment mechanics, I plan to add the agents and compare their reward structures against simpler policies. I'll examine economic growth, pollution, resource management, and the risk of environmental failure.

The goal is to better understand how independent agents make decisions when pursuing their own development can put the entire system at risk.

This is a synthetic environment designed to explore multi-agent decision-making, not to represent real economic or environmental systems.
