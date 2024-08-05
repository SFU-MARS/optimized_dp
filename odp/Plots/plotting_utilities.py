import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from odp.Grid import Grid
import numpy as np

def plot_isosurface(grid, V_ori, plot_option):
    
    dims_plot = plot_option.dims_plot

    grid, my_V = pre_plot(plot_option, grid, V_ori)

    if len(dims_plot) != 3 and len(dims_plot) != 2 and len(dims_plot) != 1:
        raise Exception('dims_plot length should be equal to 3, 2 or 1\n')

    if len(dims_plot) == 3 and len(my_V.shape) == 3:
        # Plot 3D isosurface for only one time step
        
        complex_x = complex(0, grid.pts_each_dim[0])
        complex_y = complex(0, grid.pts_each_dim[1])
        complex_z = complex(0, grid.pts_each_dim[2])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[0]:grid.max[0]: complex_x, grid.min[1]:grid.max[1]: complex_y,
                           grid.min[2]:grid.max[2]: complex_z]
        


        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V.flatten(),
            caps=dict(x_show=True, y_show=True),
            isomin=plot_option.min_isosurface,
            surface_count=plot_option.surface_count,
            isomax=plot_option.max_isosurface,
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            contour=plot_option.contour,
            flatshading=plot_option.flatshading,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition,
            reversescale=plot_option.reversescale,
            showlegend=plot_option.showlegend,
            showscale=plot_option.showscale,
        ))

    if len(dims_plot) == 3 and len(my_V.shape) == 4:
        # Plot 3D isosurface with animation
        # dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        
        complex_x = complex(0, grid.pts_each_dim[0])
        complex_y = complex(0, grid.pts_each_dim[1])
        complex_z = complex(0, grid.pts_each_dim[2])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[0]:grid.max[0]: complex_x, grid.min[1]:grid.max[1]: complex_y,
                           grid.min[2]:grid.max[2]: complex_z]

        N = my_V.shape[3]
        print("Plotting beautiful plots. Please wait\n")

        # Define frames
        fig = go.Figure(frames=[go.Frame(data = go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V[:, :, :, N-k-1].flatten(),
            caps=dict(x_show=True, y_show=True),
            isomin=plot_option.min_isosurface,
            surface_count=plot_option.surface_count,
            isomax=plot_option.max_isosurface,
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            contour=plot_option.contour,
            flatshading=plot_option.flatshading,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition,
            reversescale=plot_option.reversescale,
            showlegend=plot_option.showlegend,
            showscale=plot_option.showscale,
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(N)])

        # Add data to be displayed before animation starts
        fig.add_trace(go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V[:, :, :, N-1].flatten(),
            caps=dict(x_show=True, y_show=True),
            isomin=plot_option.min_isosurface,
            surface_count=plot_option.surface_count,
            isomax=plot_option.max_isosurface,
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            contour=plot_option.contour,
            flatshading=plot_option.flatshading,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition,
            reversescale=plot_option.reversescale,
            showlegend=plot_option.showlegend,
            showscale=plot_option.showscale,
            ))
        
        fig.update_layout(
            title='3D Set',
            scene=dict( xaxis={"nticks": 20},
                        zaxis={"nticks": 20},
                        camera_eye={"x": 0, "y": -1, "z": 0.5},
                        aspectratio={"x": 1, "y": 1, "z": 0.6}
                        ))
        
        fig = slider_define(fig)


    if len(dims_plot) == 2 and len(my_V.shape) == 2:
        # Plot 2D isosurface for only one time step
        # dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid.pts_each_dim[0])
        complex_y = complex(0, grid.pts_each_dim[1])
        mg_X, mg_Y = np.mgrid[grid.min[0]:grid.max[0]: complex_x, grid.min[1]:grid.max[1]: complex_y]


        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful 2D plots. Please wait\n")
        # TODO Chong
        fig = go.Figure(data=go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=my_V.flatten(),
            zmin=0.0,
            ncontours=1,
            contours_coloring='none',  # former: lines
            name="Reachable Set",  # zero level
            line_width=1.5,
            line_color='magenta',
            zmax=0.0,
        ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))  # ,paper_bgcolor='rgba(0,0,0,0)'

    if len(dims_plot) == 2 and len(my_V.shape) == 3:
        # Plot 2D isosurface with animation
        # dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid.pts_each_dim[0])
        complex_y = complex(0, grid.pts_each_dim[1])
        mg_X, mg_Y = np.mgrid[grid.min[0]:grid.max[0]: complex_x, grid.min[1]:grid.max[1]: complex_y]

        N = my_V.shape[2]

        print("Plotting beautiful plots. Please wait\n")
        # Define frames
        fig = go.Figure(frames=[go.Frame(data = go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=my_V[:,:,N-k-1].flatten(),
            zmin=0.0,
            ncontours=1,
            contours_coloring='none',  # former: lines
            name="Reachable Set",  # zero level
            line_width=1.5,
            line_color='magenta',
            zmax=0.0,
            ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(N)])

        # Add data to be displayed before animation starts
        fig.add_trace(go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=my_V[:,:,N-1].flatten(),
            zmin=0.0,
            ncontours=1,
            contours_coloring='none',  # former: lines
            name="Reachable Set",  # zero level
            line_width=1.5,
            line_color='magenta',
            zmax=0.0,
            ))
        
        fig.update_layout(title='2D Set',)
        
        fig = slider_define(fig)


    if len(dims_plot) == 1 and len(my_V.shape) == 1:
        # Plot 1D isosurface for only one time step
        # dim1 = dims_plot[0]
        complex_x = complex(0, grid.pts_each_dim[0])
        mg_X = np.mgrid[grid.min[0]:grid.max[0]: complex_x]

        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful 1D plots. Please wait\n")
        fig = go.Figure(data=px.line(
            x=mg_X.flatten(),
            y=my_V.flatten(),
            name="Reachable Set",
            labels={'x','Vaue'}
        ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))



    if len(dims_plot) == 1 and len(my_V.shape) == 2:
        # Plot 1D isosurface with animation
        # dim1 = dims_plot[0]
        complex_x = complex(0, grid.pts_each_dim[0])
        mg_X = np.mgrid[grid.min[0]:grid.max[0]: complex_x]
        
        N = my_V.shape[1]

        # Define frames
        fig = go.Figure(frames=[go.Frame(data=px.line(
            x=mg_X.flatten(),
            y=my_V[:,N-k-1].flatten(),
            labels={'x','Vaue'}
            ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(N)])

        # Add data to be displayed before animation starts
        fig.add_trace(go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=my_V[:,:,N-1].flatten(),
            zmin=0.0,
            ncontours=1,
            contours_coloring='none',  # former: lines
            line_width=1.5,
            line_color='magenta',
            zmax=0.0,
            ))
        
        fig.update_layout(title='1D Value Function',)
        
        fig = slider_define(fig)


    if plot_option.do_plot:
        fig.show()
        print("Please check the plot on your browser.")

    # Local figure save
    if plot_option.save_fig:
        if plot_option.interactive_html:
            fig.write_html(plot_option.filename + ".html")
        else:
            fig.write_image(plot_option.filename)


def plot_valuefunction(grid, V_ori, plot_option):
    '''
    Plot value function V, 1D or 2D grid is allowed
    https://plotly.com/python/3d-surface-plots/
    '''   
    dims_plot = plot_option.dims_plot
    grid, my_V = pre_plot(plot_option, grid, V_ori)

    if len(dims_plot) != 2 and len(dims_plot) != 1:
        raise Exception('dims_plot length should be equal to 2 or 1\n')

    if len(dims_plot) == 2 and len(my_V.shape) == 2:
        # Plot 3D surface for only one time step
        # dim1, dim2 = dims_plot[0], dims_plot[1]

        my_X = np.linspace(grid.min[0], grid.max[0], grid.pts_each_dim[0])
        my_Y = np.linspace(grid.min[1], grid.max[1], grid.pts_each_dim[1])
        my_V = my_V

        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Surface(
            # TODO chong: allow multiple sub-level sets
            contours = {
            "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
            },
            x=my_X,
            y=my_Y,
            z=my_V,
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition
            ))

    if len(dims_plot) == 2 and len(my_V.shape) == 3:
        # ref: https://plotly.com/python/visualizing-mri-volume-slices/
        # Plot 3D surface with animation
        # dim1, dim2 = dims_plot[0], dims_plot[1]
        my_X = np.linspace(grid.min[0], grid.max[0], grid.pts_each_dim[0])
        my_Y = np.linspace(grid.min[1], grid.max[1], grid.pts_each_dim[1])
        N = my_V.shape[2]

        print("Plotting beautiful plots. Please wait\n")

        # Define frames
        fig = go.Figure(frames=[go.Frame(data = go.Surface(
            # TODO chong: allow multiple sub-level sets
            contours = {
            "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
            },
            x=my_X,
            y=my_Y,
            z=my_V[:, :, N-k-1],
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition
            ),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(N)])

        # Add data to be displayed before animation starts
        fig.add_trace(go.Surface(
            # TODO chong: allow multiple sub-level sets
            contours = {
            "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
            },
            x=my_X,
            y=my_Y,
            z=my_V[:, :, N-1],
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition
            ))
        
        fig.update_layout(
            title='2D Value Function',
            scene=dict( xaxis={"nticks": 20},
                        zaxis={"nticks": 4},
                        camera_eye={"x": 0, "y": -1, "z": 0.5},
                        aspectratio={"x": 1, "y": 1, "z": 0.2}
                        ))
        
        fig = slider_define(fig)

    if len(dims_plot) == 1 and len(my_V.shape) == 1:
        # Plot 1D isosurface for only one time step
        # dim1 = dims_plot[0]
        complex_x = complex(0, grid.pts_each_dim[0])
        mg_X = np.mgrid[grid.min[0]:grid.max[0]: complex_x]


        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful 1D plots. Please wait\n")
        fig = go.Figure(data=px.line(
            x=mg_X.flatten(),
            y=my_V.flatten(),
            labels={'x','Vaue'}
        ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))

        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(range=[-1, 1.5])



    if len(dims_plot) == 1 and len(my_V.shape) == 2:
        # Plot 1D isosurface with animation
        # dim1 = dims_plot[0]
        complex_x = complex(0, grid.pts_each_dim[0])
        mg_X = np.mgrid[grid.min[0]:grid.max[0]: complex_x]
        
        N = my_V.shape[1]

        # Define frames
        fig = go.Figure(frames=[go.Frame(data=go.Scatter(
            x=mg_X.flatten(),
            y=my_V[:,N-k-1].flatten()
            ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'),
            name=str(k) # you need to name the frame for the animation to behave properly
            )
            for k in range(N)])

        # Add data to be displayed before animation starts
        fig.add_trace(go.Scatter(
            x=mg_X.flatten(),
            y=my_V[:,N-1].flatten()))
        
        fig.update_layout(title='1D Value Function',)
        
        fig = slider_define(fig, duration=0)


        fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor='black')
        fig.update_yaxes(range=[-1, 1.5])
        fig.update_layout(transition = {'duration':0})

    if plot_option.do_plot:
        fig.show()
        print("Please check the plot on your browser.")
        # Local figure save
    if plot_option.save_fig:
        if plot_option.interactive_html:
            fig.write_html(plot_option.filename + ".html")
        else:
            fig.write_image(plot_option.filename)

###################################################################################################################################
def slider_define(fig, duration=300):
    '''
    Internal function
    Define slider for the animation
    '''
    def frame_args(duration):
            return {
                    "frame": {"duration": duration},
                    "mode": "immediate",
                    "fromcurrent": True,
                    "transition": {"duration": duration},
                }
        
    sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "currentvalue": {
                    "font": {"size": 20},
                    "prefix": "Time Step:",
                    "visible": True,
                    "xanchor": "right"
                },
                "steps": [
                    {
                        "args": [[f.name], frame_args(0)],
                        "label": str(k),
                        "method": "animate",
                    }
                    for k, f in enumerate(fig.frames)
                ],
            }
        ]

        # Layout
    fig.update_layout(
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(duration)],
                                "label": "Play", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "pause", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
        )
    return fig


def pre_plot(plot_option, grid, V_ori):
    """
    Pre-processing steps for plotting
    """


    # Slicing process
    dims_plot = plot_option.dims_plot
    idx = [slice(None)] * grid.dims
    slice_idx = 0

    # Build new grid
    grid_min = grid.min
    grid_max = grid.max
    dims = grid.dims
    N = grid.pts_each_dim

    delete_idx = []
    dims_list = list(range(grid.dims))

    for i in dims_list:
        if i not in dims_plot:
            idx[i] = plot_option.slices[slice_idx]
            slice_idx += 1
            dims = dims -1
            delete_idx.append(i)
    N = np.delete(N, delete_idx)
    grid_min = np.delete(grid_min, delete_idx)
    grid_max = np.delete(grid_max, delete_idx)

    V = V_ori[tuple(idx)]
    
    grid = Grid(grid_min, grid_max, dims, N)

    # Downsamping process
    if plot_option.scale is not None:
        scale = plot_option.scale
    else:
        scale = [1] * grid.dims
        for i in range(grid.dims):
            if grid.pts_each_dim[i] > 30:
                scale[i] = np.floor(grid.pts_each_dim[i]/30).astype(int)
    grid, V = downsample(grid, V, scale)

    return grid, V

def downsample(g, data, scale):
    """
    Dowsampling for large 3D grid size, e.g. 100x100x100 for efficient plotting
    """

    if len(scale) != g.dims:
        raise Exception('scale length should be equal to grid dimension\n')

    odd_ind =[False] * g.dims
    for i in range(g.dims):
        if g.pts_each_dim[i] % scale[i] != 0:
            odd_ind[i] = True
    
    # Generate new data
    idx = [slice(0,None,scale[0])] * g.dims
    for i in range(g.dims):
        if odd_ind[i]:
                idx[i] = slice(0,-(g.pts_each_dim[i]%scale[i]),scale[i])
    data_out = data[tuple(idx)]
    # Generate new grid
    grid_min = g.min
    grid_max = g.max
    dims = g.dims
    N = g.pts_each_dim
    for i in range(g.dims):
        if odd_ind[i]:
            grid_max[i] = g.max[i]-(g.pts_each_dim[i]%scale[i])*(g.max[i]-g.min[i])/g.pts_each_dim[i]
            N[i] = ((g.pts_each_dim[i]-g.pts_each_dim[i]%scale[i])/scale[i]).astype(np.int64)
        else:
            N[i] = (g.pts_each_dim[i]/scale[i]).astype(np.int64)
    g_out = Grid(grid_min, grid_max, dims, N)

    return g_out, data_out
       