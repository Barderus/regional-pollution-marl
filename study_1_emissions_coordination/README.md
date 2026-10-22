# Regional Pollution Coordination with Multi-Agent RL

This project explores how independent agents can coordinate pollution reduction across neighboring regions using Multi-Agent Reinforcement Learning (MARL).

The idea is that each region is responsible for managing its own emissions, but pollution doesn't stay within regional boundaries. Decisions made by one region can affect its neighbors, creating a situation where what's best for one region might not be best for everyone.

I'm using this environment to explore how different reward structures influence agent behavior, particularly when agents have limited information about the rest of the system.

## The Environment

The simulation consists of four neighboring regions, each controlled by an independent Q-learning agent.

Each agent chooses between three mitigation levels based on its local observations. I'm experimenting with three reward structures:

- **Local:** Agents prioritize their own regional outcomes.
- **Global:** Agents consider the overall system.
- **Mixed:** Agents balance local and global objectives.

I'm also comparing the learned policies against simpler approaches, including random actions, fixed mitigation, and threshold-based rules.

## Current Progress

The initial experiments show that global rewards generally lead to lower pollution than local rewards, although they come with higher mitigation costs. A simple fixed-mitigation policy also performed well, which raises questions about when reinforcement learning is actually beneficial.

I've also explored pollution spillover, alternative reward designs, and how agents change their mitigation decisions under different conditions.

The notebooks and results are available in the `notebooks/` and `results/` directories.

## Next Steps

I plan to experiment with different regional networks, environmental parameters, and coordination strategies to better understand how independent agents can balance individual and shared objectives.

This is a synthetic environment designed to study decentralized decision-making, not to model or predict real-world pollution.