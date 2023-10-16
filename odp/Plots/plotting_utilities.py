import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
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

    if len(dims_plot) != 3 and len(dims_plot) != 2 and len(dims_plot) != 1:
        raise Exception('dims_plot length should be equal to 3, 2 or 1\n')

    if len(dims_plot) == 3:
        # Plot 3D isosurface
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        complex_z = complex(0, grid.pts_each_dim[dim3])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y,
                           grid.min[dim3]:grid.max[dim3]: complex_z]

        my_V = V[tuple(idx)]

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
            uid=plot_option.uid
        ))
        fig.show()
        print("Please check the plot on your browser.")

    if len(dims_plot) == 2:
        dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]

        my_V = V[tuple(idx)]

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

        fig.show()
        print("Please check the plot on your browser.")

    if len(dims_plot) == 1:
        #TODO Chong: do not need mesh grid
        dim1 = dims_plot[0]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        mg_X = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x]

        my_V = V[tuple(idx)]

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

        fig.show()
        print("Please check the plot on your browser.")


def plot_valuefunction(grid, V, plot_option):
    '''
    Plot value function V, 1D or 2D grid is allowed
    https://plotly.com/python/3d-surface-plots/
    '''
    dims_plot = plot_option.dims_plot
    idx = [slice(None)] * grid.dims
    slice_idx = 0

    dims_list = list(range(grid.dims))
    for i in dims_list:
        if i not in dims_plot:
            idx[i] = plot_option.slices[slice_idx]
            slice_idx += 1

    if len(dims_plot) != 2 and len(dims_plot) != 1:
        raise Exception('dims_plot length should be equal to 2 or 1\n')

    if len(dims_plot) == 2 and len(V.shape) == 2:
        # Plot 3D surface for only one time step
        dim1, dim2 = dims_plot[0], dims_plot[1]
        # complex_x = complex(0, grid.pts_each_dim[dim1])
        # complex_y = complex(0, grid.pts_each_dim[dim2])
        # mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]

        my_X = np.linspace(grid.min[dim1], grid.max[dim1], grid.pts_each_dim[dim1])
        my_Y = np.linspace(grid.min[dim2], grid.max[dim2], grid.pts_each_dim[dim2])
        my_V = V[tuple(idx)]

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

    if len(dims_plot) == 2 and len(V.shape) == 3:
        # Plot 3D surface with animation
        dim1, dim2 = dims_plot[0], dims_plot[1]
        # complex_x = complex(0, grid.pts_each_dim[dim1])
        # complex_y = complex(0, grid.pts_each_dim[dim2])
        # mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y]
        my_X = np.linspace(grid.min[dim1], grid.max[dim1], grid.pts_each_dim[dim1])
        my_Y = np.linspace(grid.min[dim2], grid.max[dim2], grid.pts_each_dim[dim2])
        N = V.shape[2]

        print("Plotting beautiful plots. Please wait\n")

        # make figure
        fig = go.Figure(
            data=[go.Surface(
            # TODO chong: allow multiple sub-level sets
            contours = {
            "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
            },
            x=my_X,
            y=my_Y,
            z=V[:, :, N-1],
            colorscale=plot_option.colorscale,
            opacity=plot_option.opacity,
            lighting=plot_option.lighting,
            lightposition=plot_option.lightposition
            )],
            layout=go.Layout(
                            updatemenus=[dict(type="buttons",
                                buttons=[dict(label="Play",
                                                method="animate",
                                                args=[None])])]),
            frames=[go.Frame(
                    data=go.Surface(
                    # TODO chong: allow multiple sub-level sets
                    contours = {
                    "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
                    },
                    x=my_X,
                    y=my_Y,
                    z=V[:,:,N-k-1],
                    colorscale=plot_option.colorscale,
                    opacity=plot_option.opacity,
                    lighting=plot_option.lighting,
                    lightposition=plot_option.lightposition),
                    name=str(k)
                    )

                    for k in range(N)]
            )        

        #update camera view
        fig.update_layout(scene={
                "xaxis": {"nticks": 20},
                "zaxis": {"nticks": 4},
                'camera_eye': {"x": 0, "y": -1, "z": 0.5},
                "aspectratio": {"x": 1, "y": 1, "z": 0.2}
            })
        
        #update x-axis and y-axis and hover mode
        fig.update_xaxes(range=[grid.min[dim1], grid.max[dim1]], title="x")
        fig.update_yaxes(range=[grid.min[dim2], grid.max[dim2]], title="y")
        fig.update_layout(hovermode="closest")

        #update slider
        steps = []
        for k in range(N):
            step = dict(
            method="animate",
            args=[  [str(k)],  # Verify that the frame name is passed here.
                    {"frame": {"duration": 300, "redraw": False},
                    "mode": "immediate",
                    "transition": {"duration": 300}}
                ],  # layout attribute
            label=str(k)
            )
            steps.append(step)

        sliders = [dict(
            active=0,
            yanchor = "top",
            xanchor = "left",
            currentvalue = {
                "font": {"size": 20},
                "prefix": "Time Step:",
                "visible": True,
                "xanchor": "right"
            },
            transition = {"duration": 300, "easing": "cubic-in-out"},
            pad = {"b": 10, "t": 50},
            len = 0.9,
            x = 0.1,
            y = 0,
            steps=steps
        )]

        # update button
        updatemenus = [dict(
        buttons = [
            dict(
                args = [None, {"frame": {"duration": 500, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 300,
                                                                    "easing": "quadratic-in-out"}}],
                label = "Play",
                method = "animate"
                ),
            dict(
                 args = [[None], {"frame": {"duration": 0, "redraw": False},
                                  "mode": "immediate",
                                  "transition": {"duration": 0}}],
                label = "Pause",
                method = "animate"
                )
        ],
        direction = "left",
        pad = {"r": 10, "t": 87},
        showactive = False,
        type = "buttons",
        x = 0.1,
        xanchor = "right",
        y = 0,
        yanchor = "top"
        )]

        # sliders = [dict(steps = [dict(method= 'animate',
        #                         args= [[f'frame{k}'],                           
        #                         dict(mode= 'immediate',
        #                             frame= dict(duration=400, redraw=True),
        #                             transition=dict(duration= 0))
        #                             ],
        #                         label=f'{k+1}'
        #                         ) for k in range(N)], 
        #             active=0,
        #             transition= dict(duration= 0 ),
        #             x=0, # slider starting position  
        #             y=0, 
        #             currentvalue=dict(font=dict(size=12), 
        #                             prefix='frame: ', 
        #                             visible=True, 
        #                             xanchor= 'center'
        #                             ),  
        #             len=1.0) #slider length
        #     ]

        fig.update_layout(updatemenus=updatemenus, sliders=sliders)



        # make figure
        # fig_dict = {
        #     "data": [],
        #     "layout": {},
        #     "frames": []
        # }

        # # fill in most of layout
        # fig_dict["layout"]["xaxis"] = {"range": [grid.min[dim1], grid.max[dim1]], "title": "x"}
        # fig_dict["layout"]["yaxis"] = {"range": [grid.min[dim2], grid.max[dim2]], "title": "y"}
        # fig_dict["layout"]["hovermode"] = "closest"
        # fig_dict["layout"]["updatemenus"] = [
        #     {
        #         "buttons": [
        #             {
        #                 "args": [None, {"frame": {"duration": 500, "redraw": False},
        #                                 "fromcurrent": True, "transition": {"duration": 300,
        #                                                                     "easing": "quadratic-in-out"}}],
        #                 "label": "Play",
        #                 "method": "animate"
        #             },
        #             {
        #                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
        #                                 "mode": "immediate",
        #                                 "transition": {"duration": 0}}],
        #                 "label": "Pause",
        #                 "method": "animate"
        #             }
        #         ],
        #         "direction": "left",
        #         "pad": {"r": 10, "t": 87},
        #         "showactive": False,
        #         "type": "buttons",
        #         "x": 0.1,
        #         "xanchor": "right",
        #         "y": 0,
        #         "yanchor": "top"
        #     }
        # ]

        # sliders_dict = {
        #     "active": 0,
        #     "yanchor": "top",
        #     "xanchor": "left",
        #     "currentvalue": {
        #         "font": {"size": 20},
        #         "prefix": "Time Step:",
        #         "visible": True,
        #         "xanchor": "right"
        #     },
        #     "transition": {"duration": 300, "easing": "cubic-in-out"},
        #     "pad": {"b": 10, "t": 50},
        #     "len": 0.9,
        #     "x": 0.1,
        #     "y": 0,
        #     "steps": []
        # }

        # fig_dict["layout"]["scene"] = {
        #         "xaxis": {"nticks": 20},
        #         "zaxis": {"nticks": 4},
        #         'camera_eye': {"x": 0, "y": -1, "z": 0.5},
        #         "aspectratio": {"x": 1, "y": 1, "z": 0.2}
        #     }

        # # make data
        # fig_dict["data"] = [go.Surface(
        #     # TODO chong: allow multiple sub-level sets
        #     contours = {
        #     "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
        #     },
        #     x=my_X,
        #     y=my_Y,
        #     z=V[:, :, N-1],
        #     colorscale=plot_option.colorscale,
        #     opacity=plot_option.opacity,
        #     lighting=plot_option.lighting,
        #     lightposition=plot_option.lightposition
        # )]

        # # make frames  
        # fig_dict["frames"] = [go.Frame(
        # data=go.Surface(
        # # TODO chong: allow multiple sub-level sets
        # contours = {
        # "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
        # },
        # x=my_X,
        # y=my_Y,
        # z=V[:,:,N-k-1],
        # colorscale=plot_option.colorscale,
        # opacity=plot_option.opacity,
        # lighting=plot_option.lighting,
        # lightposition=plot_option.lightposition),
        # name = str(k))
        # for k in range(N)]     

        # for k in range(N):
        #     slider_step = {
        #         "args": [
        #             [str(k)],  # Verify that the frame name is passed here.
        #             {"frame": {"duration": 300, "redraw": False},
        #             "mode": "immediate",
        #             "transition": {"duration": 300}}
        #         ],
        #         "label": str(k),  # Verify that the label matches the frame name.
        #         "method": "animate"
        #     }
        #     sliders_dict["steps"].append(slider_step)


    
        # print("I'm here")
        # fig_dict["layout"]["sliders"] = [sliders_dict]

        # fig = go.Figure(fig_dict)

         # Create figure
        # fig = go.Figure(
        #     data=[go.Surface(
        #     # TODO chong: allow multiple sub-level sets
        #     contours = {
        #     "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
        #     },
        #     x=my_X,
        #     y=my_Y,
        #     z=V[:, :, N-1],
        #     colorscale=plot_option.colorscale,
        #     opacity=plot_option.opacity,
        #     lighting=plot_option.lighting,
        #     lightposition=plot_option.lightposition
        #     )],
        #     layout=go.Layout(
        #                     updatemenus=[dict(type="buttons",
        #                         buttons=[dict(label="Play",
        #                                         method="animate",
        #                                         args=[None])])]),
        #     frames=[go.Frame(
        #             data=go.Surface(
        #             # TODO chong: allow multiple sub-level sets
        #             contours = {
        #             "z": {"show": True, "start": -1, "end": 1, "size": 1, "color":"white", },
        #             },
        #             x=my_X,
        #             y=my_Y,
        #             z=V[:,:,N-k-1],
        #             colorscale=plot_option.colorscale,
        #             opacity=plot_option.opacity,
        #             lighting=plot_option.lighting,
        #             lightposition=plot_option.lightposition)
        #             )

        #             for k in range(N)]
        #     )
        



    fig.show()
    print("Please check the plot on your browser.")

