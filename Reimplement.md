# Multi-Agent Reach-Avoid Games (MARAG)
This repository contains the implementation of the paper "Multi-Agent Reach-Avoid Games: Two Attackers Versus One Defender and Mixed Integer Programming (2023 CDC intended)". This implementation is based on the [OptimizedDP library](https://github.com/SFU-MARS/optimized_dp).

# Details to Reimplement
## Step 1: Set dynamics
The dynamics of attackers and defenders (including the maximum speed) in different reach-avoid games are in files ``AttackerDefender1v0.py``, ``AttackerDefender1v1.py`` and ``AttackerDefender2v1.py``. 

## Step 2: Generate value functions
We need to calculate several value functions for the further reach-avoid games.
* Run the file ``hjvalue1v0.py`` to obtain the 1 vs. 0 value function ``1v0AttackDefend.npy``, which is used to generate control inputs of attackers.
* Run the file ``hjvalue1v1.py`` with the grid size 45 (line 24) to generate the value fucnton ``1v1AttackDefend_speed15.npy``, which is used to generate control inputs of the defender in the 1 vs. 1 reach-avoid game:
``` python
grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # [45, 45, 45, 45], [30, 30, 30, 30]
......
np.save('/localhome/hha160/optimized_dp/MRAG/1v1AttackDefend_speed15.npy', result)  # grid = 45
```
Then run the file ``hjvalue1v1.py`` with the grid size 30 to generate the value fucntion ``1v1AttackDefend_g30_speed15.npy`` for the computation of the 2 vs. 1 reach-avoid game:
``` python
grids = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([30, 30, 30, 30])) # [45, 45, 45, 45], [30, 30, 30, 30]
......
np.save('/localhome/hha160/optimized_dp/MRAG/1v1AttackDefend_speed15.npy', result)  # grid = 45
```