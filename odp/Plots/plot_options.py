
class PlotOptions:
  def __init__(self, do_plot=True, plot_type="set", plotDims=[],
               slicesCut=[], min_isosurface = 0, max_isosurface = 0, 
               colorscale='Rainbow', save_fig=False, filename=None, interactive_html=False, contour=None,  flatshading=None,  
               legend=None, legendgroup=None, 
               legendgrouptitle=None, legendrank=None, legendwidth=None, 
               lighting=None, lightposition=None, 
               opacity=0.8, reversescale=None, 
               showlegend=None, showscale=None, figSize=None,
               surface_count=1, uid=None, scale=None):
    
    if plot_type not in ["set", "value"]:
        raise Exception("Illegal plot type !")
    
    if len(plotDims) != 1 and len(plotDims) != 2 and len(plotDims) != 3:
        raise Exception("Make sure that dim_plot size is 1, 2, or 3!!")
    
    if plot_type == "value" and len(plotDims) == 3:
        raise Exception("Make sure that dim_plot size is 1 or 2 for value function plot!!")
    
    if plot_type == "set" and len(plotDims) == 1:
        raise Exception("Make sure that dim_plot size is 2 or 3 for 0 sublevel set plot!!")

    self.do_plot = do_plot
    self.dims_plot = plotDims
    self.plot_type = plot_type
    self.slices = slicesCut
    self.min_isosurface = min_isosurface
    self.max_isosurface = max_isosurface

# Plotly save figure option
    self.save_fig = save_fig
    self.filename = filename
    self.interactive_html = interactive_html

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

    # Scale of downsampling data
    self.scale = scale







