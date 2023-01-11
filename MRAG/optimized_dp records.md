## optimized_dp records

### 1. Debug 1 for the 'not rectangle' look with the value function

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

result:

![](/localhome/hha160/optimized_dp/MRAG/debug_figures/debug0_original.png)

##### 1. to 0.7

result:

![](/localhome/hha160/optimized_dp/MRAG/debug_figures/debug1_0.7.png)

##### 2. to 0.6

result:

![](/localhome/hha160/optimized_dp/MRAG/debug_figures/debug1_0.6.png)

##### 3. to 0.5

result:

![](/localhome/hha160/optimized_dp/MRAG/debug_figures/debug1_0.5.png)

##### 4. to 0.1

result:

![](/localhome/hha160/optimized_dp/MRAG/debug_figures/debug1_0.1.png)