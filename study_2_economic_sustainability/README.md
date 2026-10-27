# Study 2: Economic Development and Environmental Sustainability

**Status: Initial comparison completed**

## Overview

This study builds on [Study 1](../study_1_emissions_coordination/README.md) by introducing economic development and resource management alongside pollution mitigation.

The idea is that each region wants to grow economically, but development also increases emissions. Regions need to decide whether to invest in their economy, spend resources on reducing pollution, or save their currency for future decisions.

The challenge is that pollution doesn't stay within regional boundaries, and if pollution in any region reaches a critical level, the entire system fails.

I'm interested in exploring how independent agents balance their own economic growth with the environmental needs of the larger system, especially when their decisions can have consequences for neighboring regions.

## The Environment

The simulation has four neighboring regions. Each can use a separate Q-learning agent with its own Q-table.

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

The environment and its settings are in `environment.py` and `config.py`. It accepts four actions at a time and returns each region's next observation and reward, separate termination and time-limit flags, and the underlying quantities used in the transition. With a reset seed, the four starting currency balances can be reproduced.

`agents.py` contains the tabular Q-learning agent. It can attempt any of the five actions; the environment decides whether an attempt is valid. The agent does not use a future-state value after environmental failure, but it does after an episode reaches the time limit. Focused checks cover the environment mechanics and learning update.

The pilot ran five simple policies over 20 matched starting seeds. Development alone caused environmental failure in all 20 episodes; the mitigation and coordination policies reached the 200-step limit in all 20. I kept the agreed environment settings for the initial reward comparison.

In the saved initial run, I trained local, global, and mixed reward agents for 2,000 episodes under each of five training seeds, then evaluated them on the same 20 held-out starts. None failed in these evaluations. Local and mixed rewards had lower average GPI than global reward, with higher mitigation spending. The fixed mitigation baseline had even lower GPI and spending, but finished with no economic development. These are outcomes under the current synthetic rules, not a general ranking of reward designs.

## Reproducing the runs

The work is organized in four notebooks:

- [00: Baselines](notebooks/00_baselines_preview.ipynb) shows each of the five fixed rules on a few starts.
- [01: Experiments](notebooks/01_experiments_preview.ipynb) shows a short Q-learning run and reward comparison.
- [02: Pilot dynamics](notebooks/02_pilot_dynamics.ipynb) checks the five rules across 20 starts and plots one trajectory.
- [03: Reward comparison](notebooks/03_reward_comparison.ipynb) trains five seeds per reward mode and evaluates each policy on 20 held-out starts.

## Next steps

I want to inspect learned action choices and whether the comparison changes across more seeds or different reward-component weights. This is a synthetic environment for studying agent behavior, not a model of real economic or environmental systems.
