# Import relevant packages
import math
import pandas as pd
import numpy as np
import numpy.ma as ma
import datetime
import geopandas as gpd
import shapely
import sys

import plotly.graph_objects as go
import contextily as cx
from tqdm import tqdm
import matplotlib as mpl
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Point, Polygon, LineString
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px

import json

from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter
from bokeh.palettes import brewer

from bokeh.io.doc import curdoc
from bokeh.models import Slider, HoverTool, Select
from bokeh.layouts import widgetbox, row, column

palette = brewer['Blues'][8]
# Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]
# Create a function the returns json_data for the month selected by the user
def json_data(selectedMonth,selectedDay):
    mn = selectedMonth
    dd = selectedDay

    gdf = hex_grid_A1.iloc[:,0:1]
    # Pull selected month from geodatafram
    gdf_mn_day = hex_grid_A1.filter(like=f"month:{selectedMonth}:-day:{selectedDay}")
    
    # Merge the GeoDataframe object (sf) with the neighborhood summary data (neighborhood)
    merged =  gpd.GeoDataFrame(pd.concat([gdf,gdf_mn_day],axis=1))
    # Bokeh uses geojson formatting, representing geographical features, with json
    mergedToLoad=merged.to_json()
    # Convert to json
    merged_json = json.loads(mergedToLoad)
    
    # Convert to json preferred string-like object 
    json_data = json.dumps(merged_json)
    return json_data

# Input geojson source that contains features for plotting for:
geosource = GeoJSONDataSource(geojson = json_data(1,1))

# Define the callback function: update_plot
def update_plot(attr, old, new):

    month = mon_slider.value
    day = day_slider.value
    new_data = json_data(month,day)
        
    # Update the plot based on the changed inputs
    p = make_plot()
    
    # Update the layout, clear the old document and display the new document
    layout = column(p, widgetbox(mon_slider), widgetbox(day_slider))
    curdoc().clear()
    curdoc().add_root(layout)
    
    # Update the data
    geosource.geojson = new_data

# Create a plotting function
def make_plot(month,day):    

  date=f"month:{month}:-day:{day}"
  # Set the format of the colorbar
  min_range = np.array(hex_grid_A1.filter(like=f"month:{month}:-day:{day}").min())[0]
  max_range = np.array(hex_grid_A1.filter(like=f"month:{month}:-day:{day}").max())[0]
  
  # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
  color_mapper = LinearColorMapper(palette = palette, low = min_range, high = max_range)

  # Create color bar.
  color_bar = ColorBar(color_mapper=color_mapper, label_standoff=18,border_line_color=None, location = (0, 0))

  # Create figure object.
  p = figure(title = f'Average number of Incidents on weekday{day} of Month{month}', 
            plot_height = 650, plot_width = 850,
            toolbar_location = None)
  p.xgrid.grid_line_color = None
  p.ygrid.grid_line_color = None
  p.axis.visible = False
  # Add patch renderer to figure. 
  p.patches('xs','ys', source = geosource, fill_color = {'field' : date, 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
  
  # Specify color bar layout.
  p.add_layout(color_bar, 'right')

  # Add the hover tool to the graph
  return p

# Call the plotting function
p = make_plot(1,1)

# Make a slider object: slider 
mon_slider = Slider(title = 'Month',start = 1, end = 12, step = 1, value = 1)
mon_slider.on_change('value', update_plot)
# Make a slider object: slider 
day_slider = Slider(title = 'Weekday',start = 1, end = 7, step = 1, value = 1)
day_slider.on_change('value', update_plot)

# Make a column layout of widgetbox(slider) and plot, and add it to the current document
# Display the current document
layout = column(p, widgetbox(mon_slider), widgetbox(day_slider))
curdoc().add_root(layout)
