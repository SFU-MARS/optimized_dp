import plotly.graph_objects as go
from plotly.graph_objects import Layout
import numpy as np


def plot_isosurface(grid, V, plot_option):
    dims_plot = plot_option.dims_plot
    idx = [slice(None)] * grid.dims
    slice_idx = 0

    dims_list = list(range(grid.dims))
    for i in dims_list:
        if i not in dims_plot:
            idx[i] = plot_option.slices[slice_idx]
            slice_idx += 1

    if len(dims_plot) == 2:
        dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        mg_X, mg_Y= np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
        print("Plotting beautiful 2D plots. Please wait\n")
        fig = go.Figure(data=go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=V.flatten(),
            colorscale='jet'
        ))
        fig.show()
        print("Please check the plot on your browser.")

    elif len(dims_plot) == 3:
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        complex_z = complex(0, grid.pts_each_dim[dim3])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y,
                           grid.min[dim3]:grid.max[dim3]: complex_z]

        my_V = V[tuple(idx)]

        if (V > 0.0).all() or (V < 0.0).all():
            print("Implicit surface will not be shown since all values have the same sign ")
        print("Plotting beautiful 3D plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V.flatten(),
            colorscale='jet',
            isomin=plot_option.min_isosurface,
            surface_count=1,
            isomax=plot_option.max_isosurface,
            caps=dict(x_show=True, y_show=True)
        ))
        fig.show()
        print("Please check the plot on your browser.")

    else:
        raise Exception('dims_plot length should be equal to 2 or 3\n')


def plot_2d(grid, V_2D):
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
        zmin=0.0,
        ncontours=1,
        contours_coloring = 'lines',
        line_width = 1.5,
        line_color = 'Red',
        zmax=0.0,
    ), layout=Layout(plot_bgcolor='rgba(0,0,0,0)')) #,paper_bgcolor='rgba(0,0,0,0)'
    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=2.0))
    fig.add_shape(type='line', x0=-0.1, y0=-1.0, x1=-0.1, y1=-0.3, line=dict(color='black', width=2.0))
    fig.add_shape(type='line', x0=0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    fig.add_shape(type='line', x0=-0.1, y0=-0.3, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=2.0))
    # figure settings
    fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0), paper_bgcolor="White") # LightSteelBlue
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")


def plot_2d_with_avoid(grid, V_2D):
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
    x_obstacle = np.linspace(-0.5, 0.5, num=mg_X.flatten().shape[0])
    y_obstacle = np.linspace(-0.5, 0.5, num=mg_X.flatten().shape[0])
    V_obstacle = np.ones(V_2D.shape)
    # print(f'The shape of mg_X before flatten is {mg_X.shape}')
    # print(f'The shape of mg_Y before flatten is {mg_Y.shape}')
    # print(f'The shape of V_2D before flatten is {V_2D.shape}')
    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
        zmin=0.0,
        ncontours=1,
        zmax=0.0,
    ))
    fig.add_trace(go.Contour(
        x=x_obstacle.flatten(),
        y=y_obstacle.flatten(),
        z=V_obstacle.flatten(),
        zmin=0.0,
        ncontours=1,
        zmax=0.0,
    ))
    fig.show()
    # print(f'The shape of x after flatten is {mg_X.flatten().shape}')
    # print(f'The shape of y after flatten is {mg_Y.flatten().shape}')
    # print(f'The shape of z after flatten is {V_2D.flatten().shape}')
    print("Please check the plot on your browser.")

def plot_game(grid, V_2D, attackers, defenders):
    # based on the plot_2d, add the attacker and the defender 
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
    x_attackers = [a[0] for a in attackers]
    y_attackers = [a[1] for a in attackers]
    x_defenders = [d[0] for d in defenders]
    y_defenders = [d[1] for d in defenders]
    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
        zmin=0.0,
        ncontours=1,
        contours_coloring = 'none', # former: lines 
        name= "Reachable Set", # zero level
        line_width = 1.5,
        line_color = 'magenta',
        zmax=0.0,
    ), layout=Layout(plot_bgcolor='rgba(0,0,0,0)')) #,paper_bgcolor='rgba(0,0,0,0)'
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=2.0), name="Target")
    fig.add_trace(go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')))
    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=2.0))
    fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))
    # fig.add_shape(type='line', x0=-0.1, y0=-1.0, x1=-0.1, y1=-0.3, line=dict(color='black', width=2.0))
    # fig.add_shape(type='line', x0=0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    # fig.add_shape(type='line', x0=-0.1, y0=-0.3, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    # plot attackers
    fig.add_trace(go.Scatter(x=x_attackers, y=y_attackers, mode="markers", name='Attacker', marker=dict(symbol="triangle-up", size=10, color='red')))
    # plot defenders
    fig.add_trace(go.Scatter(x=x_defenders, y=y_defenders, mode="markers", name='Defender', marker=dict(symbol="square", size=10, color='blue')))
   
    # figure settings
    fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0), paper_bgcolor="White") # LightSteelBlue
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")

def plot_original(grid, V_2D):
    dims_plot = [0, 1]
    dim1, dim2 = dims_plot[0], dims_plot[1]
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
    print("Plotting beautiful 2D plots. Please wait\n")
    fig = go.Figure(data=go.Contour(
        x=mg_X.flatten(),
        y=mg_Y.flatten(),
        z=V_2D.flatten(),
        zmin=0.0,
        ncontours=1,
        zmax=0.0,
    ))
    fig.show()
    print("Please check the plot on your browser.")

def plot_simulation(attackers_x, attackers_y, defenders_x, defenders_y):

    print("Plotting beautiful 2D plots. Please wait\n")

    fig = go.Figure(data = go.Scatter(x=[0.6, 0.8], y=[0.1, 0.1], mode='lines', name='Target', line=dict(color='purple')), 
                    layout=Layout(plot_bgcolor='rgba(0,0,0,0)')) # for the legend
    # plot target
    fig.add_shape(type='rect', x0=0.6, y0=0.1, x1=0.8, y1=0.3, line=dict(color='purple', width=2.0), name="Target")

    # plot obstacles
    fig.add_shape(type='rect', x0=-0.1, y0=0.3, x1=0.1, y1=0.6, line=dict(color='black', width=2.0), name="Obstacle")
    fig.add_shape(type='rect', x0=-0.1, y0=-1.0, x1=0.1, y1=-0.3, line=dict(color='black', width=2.0))
    fig.add_trace(go.Scatter(x=[-0.1, 0.1], y=[0.3, 0.3], mode='lines', name='Obstacle', line=dict(color='black')))
    
    # plot attackers
    # fig.add_trace(go.Scatter(x=attackers_x[0], y=attackers_y[0], mode="lines+markers", name="Attacker", marker=dict(symbol="triangle-up", size=10, color='red')))

    for i in range(len(attackers_x)):
        fig.add_trace(go.Scatter(x=attackers_x[i], y=attackers_y[i], mode="lines+markers", name=f"Attacker{i+1}", marker=dict(symbol="triangle-up", size=3, color='red')))
    # plot defenders
    for j in range(len(defenders_x)):
        fig.add_trace(go.Scatter(x=defenders_x[j], y=defenders_y[j], mode="lines+markers", name=f'Defender{j+1}', marker=dict(symbol="square", size=3, color='blue')))

    # figure settings
    fig.update_layout(autosize=False, width=500, height=500, margin=dict(l=50, r=50, b=100, t=100, pad=0), paper_bgcolor="White", xaxis_range=[-1, 1], yaxis_range=[-1, 1]) # LightSteelBlue
    fig.update_xaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False
    fig.update_yaxes(showline = True, linecolor = 'black', linewidth = 1.0, griddash = 'dot', zeroline=False, gridcolor = 'Lightgrey', mirror=True, ticks='outside') # showgrid=False,
    fig.show()
    print("Please check the plot on your browser.")