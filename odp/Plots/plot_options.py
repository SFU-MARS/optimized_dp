
class PlotOptions:
  def __init__(self, do_plot=True, plot_type="3d_plot", plotDims=[],
               slicesCut=[], min_isosurface = 0, max_isosurface = 0, 
               colorscale='Rainbow', contour=None,  flatshading=None,  
               legend=None, legendgroup=None, 
               legendgrouptitle=None, legendrank=None, legendwidth=None, 
               lighting=None, lightposition=None, 
               opacity=0.8, reversescale=None, 
               showlegend=None, showscale=None, 
               surface_count=1, uid=None):
    
    if plot_type not in ["2d_plot", "3d_plot"]:
        raise Exception("Illegal plot type !")

    if plot_type == "2d_plot" :
        if len(plotDims) != 2:
            raise Exception("Make sure that dim_plot size is 2 !!")

    if plot_type == "3d_plot" :
        if len(plotDims) != 3:
            raise Exception("Make sure that dim_plot size is 3 !!")

    self.do_plot = do_plot
    self.dims_plot = plotDims
    self.plot_type = plot_type
    self.slices = slicesCut
    self.min_isosurface = min_isosurface
    self.max_isosurface = max_isosurface

# Editable Visualization  (Plotly)
# https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Isosurface.html
# https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.contour.html?highlight=contour#module-plotly.graph_objects.contour
    
    
    self.colorscale = colorscale
    # can be self-defined with a mapping for the lowest (0) and the highest (1).
    # eg. [[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']]
    # Alternatively, a palette name string of the follwoing list:
    # Blackbody,Bluered,Blues,C ividis,Earth,Electric,Greens,Greys,Hot,Jet,
    # Picnic,Portl and,Rainbow,RdBu,Reds,Viridis,YlGnBu,YlOrRd  

    self.contour = contour
    self.flatshading = flatshading

    # TODO: Legends 
    self.legend = legend
    self.legendgroup = legendgroup 
    self.legendgrouptitle = legendgrouptitle
    self.legendrank = legendrank
    self.legendwidth = legendwidth

    # Lights
    self.lighting = lighting
    self.lightposition = lightposition

    self.opacity = opacity
    self.reversescale = reversescale

    self.showlegend = showlegend
    self.showscale = showscale

    # Sets the number of iso-surfaces between minimum and maximum iso-values
    self.surface_count = surface_count

    # Assign an id to this trace
    self.uid = uid







