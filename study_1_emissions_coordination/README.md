# Regional Pollution Coordination with Multi-Agent RL

## Motivation

One region's emissions can affect its neighbors, so a good decision for one region may be a poor one for the group. I built this simulation to see how the agents' rewards affect their mitigation decisions.

## Method

The simulation has four regions arranged in a square. Each region chooses low, medium, or high mitigation based only on its own demand category and pollution bin. Pollution persists over time and spreads to neighboring regions.

I trained independent tabular Q-learning agents with local, global, and mixed rewards. I compared them with random actions, fixed-medium mitigation, and simple threshold rules. The results use 20 training seeds, evaluated on the same 20 demand sequences. I report pollution and mitigation cost separately. Cost is a penalty in the simulation, not money.

## Reward policies

Local, global, and mixed describe the reward each learner receives, not different ways of selecting actions. In every condition, each region has its own Q-table and chooses low, medium, or high mitigation from its own demand category and pollution bin. Local reward scores that region's pollution and action cost. Global reward is the average of the four regions' local rewards, and every learner receives that same value. Mixed reward gives each learner 70% of its own local reward and 30% of the global average.

During training, agents sometimes explore by choosing a random action; otherwise they choose the action with the best current Q-value. During evaluation, they choose the best-valued action without exploration. The random, fixed-medium, and threshold policies are hand-written baselines, not trained agents.

## Results

With neighbor spillover set to 0.2, agents trained with a global reward averaged **24.97** regional pollution and **757** mitigation cost per episode. With a local reward, they averaged **26.18** pollution and **527** cost. Global rewards reduced pollution in 18 of 20 paired training runs, but cost more in all 20.

Fixed-medium mitigation had the lowest average pollution in this sample (**23.88**), at a cost of **800**. At similar costs, mixtures of simple threshold rules matched or improved on the learned policies' pollution results. The reward choice changed the balance between pollution and cost, and learning did not consistently beat the simpler rules.

I also tried removing spillover and adding a reward term for reducing harm to neighbors. Removing spillover did not clearly change the pollution gap between global and local rewards. The neighbor-harm reward sometimes reduced pollution, usually with higher mitigation cost, but results varied substantially between runs.

## Limitations

The study uses arbitrary units, fixed assumptions, one four-region square network, three mitigation actions, and 50-step episodes. Agents see only their own demand category and a coarse pollution bin; they do not see their neighbors' states or actions. The results cover the tested seeds and settings, not other network layouts, parameter values, algorithms, or real pollution systems. They do not support policy decisions. The notebooks and saved results are in `notebooks/` and `results/`.
