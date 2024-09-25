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
import matplotlib.pyplot as plt
from matplotlib import cm
from shapely.geometry import Point, Polygon, LineString
import os
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.express as px

import json
import pyogrio

from bokeh.io       import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models   import GeoJSONDataSource, LinearColorMapper, ColorBar, RadioButtonGroup,Slider
from bokeh.palettes import brewer

from bokeh.io.doc import curdoc
from bokeh.layouts import widgetbox, row, column


# 0) Call saved hexagons as geodataframes
#hexagons_file_path = f'saved_tables/{regios_dict[regionCode]}Hexagons.json'  
#with open(hexagons_file_path, 'r') as f:
#    geojson_string = f.read()
#geojson_string2 = json.loads(geojson_string)
#geojson_data1 = json.loads(geojson_string2)
#GDFhexagons = gpd.GeoDataFrame.from_features(geojson_data1["features"])

# 1) Define Regions dictionary and Call saved geotables as geodataframes
#regionCode = int(input("Insert 1 for FLGV and 2 for Twente:_____"))
regios_dict = {1:"FGV",2:"Twente"}

#hexgrid1 = gpd.read_file(f'saved_tables/hexgrid{regios_dict[regionCode]}_2122_A1.geojson')
#hexgrid2 = gpd.read_file(f'saved_tables/hexgrid{regios_dict[regionCode]}_2122_A2.geojson')
#hexgrids = [hexgrid1,hexgrid2]

# 2) Define date labels and sequential multi-hue color palette and reverse color order so that dark blue is highest obesity.
Day_Labels = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
Month_Labels= ["January","February","March","April","May","June","July","August","September","October","November","December"]
palette = brewer['Blues'][8]
palette = palette[::-1]

# 3) Function that returns json_data for the region+month+weekday+urgency selected 
def json_data(selectedRegion,selectedMonth,selectedDay,selectedUrgency):
    # convert from A 1/2 to 0/1 index
    # ur = selectedUrgency-1    
    # Pick region's table using selectedRegion and urgency
    urg_gdf = gpd.read_file(f'saved_tables/hexgrid{regios_dict[selectedRegion]}_2122_A{selectedUrgency}.geojson')
    # Select urgencyGrid 
    # urg_gdf = hexgrids[int(ur)]
    # Pull selected month+day from geodataframe into df
    gdf_mn_day = urg_gdf.filter(like=f"month:{selectedMonth}:-day:{selectedDay}")
    # Merge the hexagons (gdf) with the data (gdf_mn_day)
    merged =  gpd.GeoDataFrame(pd.concat([gpd.GeoDataFrame(urg_gdf.geometry),gdf_mn_day],axis=1))
    # Bokeh uses geojson formatting, representing geographical features, with json
    mergedToLoad=merged.to_json()
    # Convert to json
    merged_json = json.loads(mergedToLoad)
    # Convert to json preferred string-like object 
    json_data = json.dumps(merged_json)
    return json_data
# Input geojson source that contains features for starting plot when opening application
geosource = GeoJSONDataSource(geojson = json_data(1,1,1,1))

# 4) Define the callback function: update_plot which obv updates plots with values selcted on applet
def update_plot(attr, old, new):
    region   = rad_regio.active + 1
    month    = mon_slider.value
    day      = rad_group.active + 1
    urgency  = urg_slider.value
    new_data = json_data(region,month,day,urgency)        
    # Update the plot based on the changed inputs
    p = make_plot(region,month,day,urgency)
    # Update the layout, clear the old document and display the new document
    sliders = column(rad_regio,mon_slider,urg_slider,rad_group)
    layout = row(p, sliders)
    curdoc().clear()
    curdoc().add_root(layout)
    # Update the data
    geosource.geojson = new_data

# 5) Create a plotting function which defines what the plot looks like 
def make_plot(region,month,day,urgency):    
  
  date=f"month:{month}:-day:{day}"
  # urg = urgency - 1
  urg_gdf = gpd.read_file(f'saved_tables/hexgrid{regios_dict[region]}_2122_A{urgency}.geojson')
  # Set the format of the colorbar
  min_range = np.array(urg_gdf.filter(like=f"month:{month}:-day:{day}").min())[0]
  max_range = np.array(urg_gdf.filter(like=f"month:{month}:-day:{day}").max())[0]
  # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
  color_mapper = LinearColorMapper(palette = palette, low = min_range, high = max_range)
  # Create color bar.
  color_bar = ColorBar(color_mapper=color_mapper, label_standoff=18,border_line_color=None, location = (0, 0))

  # Create figure object.
  p = figure(title = f"Average number of A{urgency} incidents on {Day_Labels[day-1]}\'s of {Month_Labels[month-1]} in {regios_dict[region]}", 
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
  return p

# 6) Call the plotting function 
p = make_plot(1,1,1,1)

# 7) Add checkbox group for weekdays. 
rad_group = RadioButtonGroup(labels=Day_Labels, active=0)
rad_group.on_change('active', update_plot) # rad_group returns [i,j] if i,j clicked, otherwise [].

# 8) Add checkbox group for regions (trial). 
rad_regio = RadioButtonGroup(labels=list(regios_dict.values()), active=0)
rad_regio.on_change('active', update_plot)
# Make a MONTHS slider object 
mon_slider = Slider(title = 'Month',start = 1, end = 12, step = 1, value = 1)
mon_slider.on_change('value', update_plot)
# Make a URGENCY slider object 
urg_slider = Slider(title = 'Urgency A',start = 1, end = 2, step = 1, value = 1)
urg_slider.on_change('value', update_plot)

# 9) Make a column layout of widgetbox(slider) and plot, and add it to the current document
# Display the current document
sliders = column(mon_slider,urg_slider,rad_group)
layout = row(p,sliders)
#layout = column(p, widgetbox(mon_slider), widgetbox(day_slider), widgetbox(urg_slider))
curdoc().add_root(layout)