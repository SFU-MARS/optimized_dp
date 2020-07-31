import plotly.graph_objects as go
import numpy as np

def plot_isosurface(grid, V, dims_plot):
    if len(dims_plot) != 3:
        raise Exception('dims_plot length should be equal to 3\n')
    else:
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        complex_z = complex(0, grid.pts_each_dim[dim3])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y,
                                      grid.min[dim3]:grid.max[dim3]: complex_z]

        # Hardcode this number for now
        # V = V[:, :, 49, :] # DubinsCar4D

        # V = V[:, :, :, 0, 0] # RelDyn5D
        V = V[:, :, :, 34, 0]  # RelDyn5D
        # V = V[:, :, :, 34, 17]  # RelDyn5D
        # V = V[:, :, :, 34, 34]  # RelDyn5D


        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=V.flatten(),
            colorscale='jet',
            isomin=0,
            surface_count=1,
            isomax=0,
            caps=dict(x_show=True, y_show=True)
        ))
        fig.update_layout(
            title="t = 2.0s, v_human = 17m/s, v_robot = 17m/s, a_human in [-5, 3], w_human in [-pi/6, pi/6]"
            # legend_title="Legend Title",
            # font=dict(
            #     family="Courier New, monospace",
            #     size=18,
            #     color="RebeccaPurple"
            # )
        )

        fig.show()
        # fig.write_image("/Users/anjianli/Desktop/robotics/project/prediction-reachability/reachable-set/t5_vh_8.5_vr_8.5_ah_-5_-2_wh_-0.1_0.1.png")
        print("Please check the plot on your browser.")

