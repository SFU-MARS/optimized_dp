import numpy as np

# RA1v2C = {}
# for i in range(2):
#     RA1v2C[]
import plotly.graph_objects as go

# Sample data for demonstration purposes
attacker_trajs = [[(0, 0), (1, 1), (3, 4)], [(2, 1), (5, 6), (7, 8)]]

# Initialize figure
fig = go.Figure()

# Add each attacker's trajectories
for idx, traj in enumerate(attacker_trajs):
    x_vals = [point[0] for point in traj]
    y_vals = [point[1] for point in traj]
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Attacker {idx+1}'))

# Update layout
fig.update_layout(title='Attackers Trajectories',
                  xaxis_title='X Position', yaxis_title='Y Position')

# Show plot
fig.show()
