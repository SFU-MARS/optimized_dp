## optimized_dp records

### Debug 1 for the 'not rectangle' look with the value function

#### Change the value (https://github.com/Hu-Hanyang/optimized_dp/blob/6cbec48660659df6ab13f59d50367cd01f29d60a/odp/computeGraphs/graph_4D.py#L52) from 0.8 

##### 0. Original setup and result

value function calculation setting:

```python
g = Grid(np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0]), 4, np.array([45, 45, 45, 45])) # original 45,; 80 doesn't work
avoid_set = np.minimum(obs3_capture, np.minimum(obs1_attack, obs2_attack)) # original
reach_set = np.minimum(np.maximum(goal1_destination, goal2_escape), np.minimum(obs1_defend, obs2_defend)) # original
lookback_length = 5.0  # try 1.5, 2.0, 2.5, 3.0, 5.0, 6.0, 8.0
t_step = 0.05
```

plotting setting:

```python
x_d = -0.3
y_d = 0.5
x_defender, y_defender = loca2slices(x_location=x_d, y_location=y_d, slices=45)
print(f'The defender is at the location [{x_d}, {y_d}] \n')
V_2D = value_function[:, :, x_defender, y_defender, 0]  # 0 is reachable set, -1 is target set
plot_2d(g, V_2D=V_2D)
```

### Bug 1 - Solved
It seems that the bug is caused by the dynamics. The control (both attacker and defender) could be zero sometimes. Adding the zero-judgement logic, the problem seems solved.

### 2. Records of the simulation
#### 2v1 reach-avod game
1. Config1
attackers_initials = [(0.0, 0.0), (0.0, 0.8)] 
defenders_initials = [(0.3, 0.5)]
result:
The current status at iteration118 of attackers is arrived:[0, 0] + been captured:[1, 1]. 
All attackers have arrived or been captured at the time t=0.595. 

2. Config2
attackers_initials = [(-0.5, 0.0), (0.0, 0.8)] 
defenders_initials = [(0.3, -0.3)]
result:
The current status at iteration172 of attackers is arrived:[0, 0] + been captured:[1, 1]. 
All attackers have arrived or been captured at the time t=0.865. 

3. Config3
attackers_initials = [(-0.5, 0.5), (-0.3, -0.8)] 
defenders_initials = [(0.0, 0.0)]
result:
The current status at iteration211 of attackers is arrived:[0, 0] + been captured:[1, 1]. 
All attackers have arrived or been captured at the time t=1.06. 