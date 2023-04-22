# Multi-Agent Reach-Avoid Games (MARAG)
This repository contains implementation of the paper "Multi-Agent Reach-Avoid Games: Two Attackers Versus One Defender and Mixed Integer Programming (2023 CDC intended)" based on the [OptimizedDP library](https://github.com/SFU-MARS/optimized_dp).

# Details to Reimplement
## Step 0: Generate value functions
We need to calculate several value functions for the further reach-avoid games.
* Run the file ``hjvalue1v0.py`` to obtain the 1 vs. 0 value function ``1v0AttackDefend.npy``, which is used to generate control inputs of attackers.
* 