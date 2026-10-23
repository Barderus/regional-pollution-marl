# Regional Pollution Coordination with Multi-Agent RL

## Motivation

One region's emissions can affect its neighbors, so a good decision for one region may be a poor one for the group. I built this simulation to see how the agents' rewards affect their mitigation decisions.

## Method

The simulation has four regions arranged in a square. Each region chooses low, medium, or high mitigation based only on its own demand category and pollution bin. Pollution persists over time and spreads to neighboring regions.

I trained independent tabular Q-learning agents with local, global, and mixed rewards. I compared them with random actions, fixed-medium mitigation, and simple threshold rules. The results use 20 training seeds, evaluated on the same 20 demand sequences. I report pollution and mitigation cost separately. Cost is a penalty in the simulation, not money.

## Results

With neighbor spillover set to 0.2, agents trained with a global reward averaged **24.97** regional pollution and **757** mitigation cost per episode. With a local reward, they averaged **26.18** pollution and **527** cost. Global rewards reduced pollution in 18 of 20 paired training runs, but cost more in all 20.

Fixed-medium mitigation had the lowest average pollution in this sample (**23.88**), at a cost of **800**. At similar costs, mixtures of simple threshold rules matched or improved on the learned policies' pollution results. The reward choice changed the balance between pollution and cost, and learning did not consistently beat the simpler rules.

I also tried removing spillover and adding a reward term for reducing harm to neighbors. Removing spillover did not clearly change the pollution gap between global and local rewards. The neighbor-harm reward sometimes reduced pollution, usually with higher mitigation cost, but results varied substantially between runs.

## Limitations

The simulation uses arbitrary units, fixed assumptions, one four-region network, and limited information for each agent. It helps study how agents learn in this setting; it does not model real pollution or support policy decisions. Other network layouts and parameter values are still open questions. The notebooks and saved results are in `notebooks/` and `results/`.
