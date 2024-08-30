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
# Import data on 1) gemeenten
geemapNL=gpd.read_file(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/geodata/gadm41_NLD_2.json')
geemapNLL=geemapNL.loc[geemapNL.ENGTYPE_2=='Municipality'] #excludes waterbodies

# 2) GGd regions
ggdmap=gpd.read_file(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/geodata/GGD_Regiogrenzen.json')

# 3) stations
stationinfo=pd.read_excel(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/regiosdata/fgm/FLGVregio_info.xlsx','Station FLGV')
geostat=np.array(stationinfo[1:])[:,2:]
statdf=pd.DataFrame({'Station':geostat[:,0].tolist(),'Latitude':geostat[:,1].tolist(),'Longitude':geostat[:,2].tolist()})
# turn into gdf
statgdf=gpd.GeoDataFrame(statdf, geometry=gpd.points_from_xy(statdf.Longitude, statdf.Latitude))
statgdf.crs='epsg:4326'
#statgdf=statgdf.to_crs(epsg=3857) converts geometry to dutch grid

# List of all municiplaities
fldgemten=[]
for i in range(len(geemapNL)):
    if geemapNL['NAME_1'][i]=='Flevoland':
        fldgemten.append(geemapNL['NAME_2'][i])
gvs_gemten=['Blaricum','GooiseMeren','Hilversum','Huizen','Laren','Weesp','Wijdemeren']
flgv_gemten=fldgemten+gvs_gemten 

# Map of the FGV geemente
geefgv=geemapNL.loc[[geemapNL['NAME_2'][i] in flgv_gemten for i in range(len(geemapNL))]]
geefgv['geometry']=geefgv.translate(xoff=-0.005)

rest_map=ggdmap.loc[(ggdmap['statnaam']!='GGD Flevoland') & (ggdmap['statnaam']!='GGD Gooi en Vechtstreek')]

# Read in data for GGD locally. Maybe implement it from link (sharepoint)
data_fld=pd.read_excel(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/regiosdata/fgm/FLD_08_21-09_22.xlsx')
data_gvs=pd.read_excel(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/regiosdata/fgm/GVS_08_21-09_22.xlsx')
flgv_post=np.vstack((np.array(data_fld)[:,2:],np.array(data_gvs)[:,2:]))
flgv_post=np.delete(flgv_post,[2,3,4,5,6,7,8,12,13,14,15,16,17,18,19,20,21,22],axis=1)

# Clean dataset from invalid runs
invalid_timestamps=[]
for i in range(len(flgv_post)):
    if (flgv_post[i,1]-flgv_post[i,0]).total_seconds()>3600: #time delta between call and ride assignment toolong 'f >1h
        invalid_timestamps.append(i)
flgv_post = pd.DataFrame(np.delete(flgv_post,invalid_timestamps,axis=0)) #only records correct calls regardeless of ridetype
flgv_post.rename(columns={2:'Station',3:'xcoord',4:'ycoord',5:'urg',0:'CallTime'},inplace=True)

# Select incidents that happened inside the region -- GFV here
gdf = gpd.GeoDataFrame(flgv_post, geometry=gpd.points_from_xy(flgv_post.xcoord, flgv_post.ycoord))
gdf.crs='epsg:28992'
gdf=gdf.to_crs(epsg=4326)
gdf_within_fgv = gdf.loc[gdf.within(geefgv.geometry.unary_union)].reset_index().drop('index',axis=1)
gdf_within_fgv = gdf_within_fgv.assign(lon = gdf_within_fgv.geometry.x.to_list())
gdf_within_fgv= gdf_within_fgv.assign(lat = gdf_within_fgv.geometry.y.to_list())

# Read in hexagon centres from local file -- maybe implement read directly from link/sharepoint.
hex_centres = [None]*4 # Initialize a list with 4 None elements
for i, region in tqdm(enumerate(['Flevoland Gooi Vecht (FGM)','Zuid Holland Zuid (ZHZ)','Twente','IJsselland'])):
    hex_centres[i] = pd.read_excel(r'C:/Users/MC/OneDrive - Stokhos BV/Stokhos/geodata/Hex_coords.xlsx',region).drop('Unnamed: 0',axis=1).drop(0).reset_index().drop('index',axis=1)
    hex_centres[i] = hex_centres[i].rename(columns={hex_centres[i].columns[1]: "lon", hex_centres[i].columns[0]: "lat"})
    hex_centres[i] = hex_centres[i].reindex(columns=hex_centres[i].columns[::-1])
    hex_centres[i].sort_values(by=['lon', 'lat']).reset_index().drop('index',axis=1)
    hex_centres[i]['hexID'] = hex_centres[i].reset_index().index

def make_hexagons(center_coords):

    # Define the size of the hexagons    
    avg_distance=np.mean([center_coords.lon.loc[i+1]-center_coords.lon.loc[i+0] for i in range(10) if (center_coords.lon.loc[i+1]-center_coords.lon.loc[i])>0])
    rad_km = avg_distance / np.sqrt(3) * 111.320 * np.cos(math.radians(center_coords.loc[5].lat))

    hexagons = []
    hex_vertices = []
    
    for j in range(len(center_coords)):
        
        center_km = [center_coords.loc[j].lon * 111.320 * np.cos(math.radians(center_coords.loc[j].lat)) , center_coords.loc[j].lat * 110.574]
        vertices = []
        
        for i in range(6):

            angle = np.pi/6 + np.pi/3 * i
            # find new vertices coords and immediately convert back to deg 
            ver_lon = (center_km[0] + rad_km * np.cos(angle)) / (111.320 * np.cos(math.radians(center_coords.loc[j].lat)))  
            ver_lat = (center_km[1] + rad_km * np.sin(angle)) / 110.574
            vertices.append((ver_lon,ver_lat))

        hexagons.append(Polygon(vertices))
    
    return hexagons

# Create two empty grids (geo-dataframes) to fill with A1 and A2 data 
hex_grid_A1=gpd.GeoDataFrame({'geometry':make_hexagons(hex_centres[0])})
hex_grid_A1.crs='epsg:4326'

hex_grid_A2=gpd.GeoDataFrame({'geometry':make_hexagons(hex_centres[0])})
hex_grid_A2.crs='epsg:4326'

hex_grids=[hex_grid_A1,hex_grid_A2]

test_months= int(input("how many test months?"))
test_days  = int(input("how many test days?"))
# fill in grids with incidents in each hexagon by month and weekday (12 x 7 entries)
for i,urg in enumerate(['A1','A2']):
    hex_grid=hex_grids[i]
    for month in tqdm(range(1,int(test_months)+1)):
        for weekday in range(test_days):
            ridesMonthDay = gdf_within_fgv[(gdf_within_fgv['urg']==f'{urg}') & (gdf_within_fgv['CallTime'].dt.weekday==weekday)  & (gdf_within_fgv['CallTime'].dt.month==month)]
            # normalisation is number of mondays,sundays ets in any given month.
            normalisation = len(np.unique(gdf_within_fgv[(gdf_within_fgv['CallTime'].dt.weekday==weekday)  & (gdf_within_fgv['CallTime'].dt.month==month)].CallTime.dt.date))            
            if normalisation==0: 
                normalisation=1
            # in following line rides are assigned to hexagons
            column_to_insert = [ridesMonthDay.within(hex_grid.loc[i].geometry).sum()/normalisation for i in range(len(hex_grid))]
            hex_grid.insert(np.shape(hex_grid)[1],f"month:{month}:-day:{weekday+1}",column_to_insert)

# Define a sequential multi-hue color palette.
palette = brewer['Blues'][8]
# Reverse color order so that dark blue is highest obesity.
palette = palette[::-1]
# Create a function the returns json_data for the month+weekday+urgency selected by the user
def json_data(selectedMonth,selectedDay,selectedUrgency):
    ur = selectedUrgency-1
    # Select urgencyGrid 
    urg_gdf = hex_grids[int(ur)]
    # Pull selected month+day from geodataframe into df
    gdf_mn_day = urg_gdf.filter(like=f"month:{selectedMonth}:-day:{selectedDay}")
    # Retrieve geofeatures from gdf
    gdf = urg_gdf.iloc[:,0:1]
    # Merge the GeoDataframe object (gdf) with the data (gdf_mn_day)
    merged =  gpd.GeoDataFrame(pd.concat([gdf,gdf_mn_day],axis=1))
    # Bokeh uses geojson formatting, representing geographical features, with json
    mergedToLoad=merged.to_json()
    # Convert to json
    merged_json = json.loads(mergedToLoad)
    
    # Convert to json preferred string-like object 
    json_data = json.dumps(merged_json)
    return json_data

# Input geojson source that contains features for plotting for:
geosource = GeoJSONDataSource(geojson = json_data(1,1,2))

# Define the callback function: update_plot
def update_plot(attr, old, new):

    month = mon_slider.value
    day = day_slider.value
    urgency = urg_slider.value
    new_data = json_data(month,day,urgency)
        
    # Update the plot based on the changed inputs
    p = make_plot(month,day,urgency)
    
    # Update the layout, clear the old document and display the new document
    layout = column(p, widgetbox(mon_slider), widgetbox(day_slider), widgetbox(urg_slider))
    curdoc().clear()
    curdoc().add_root(layout)
    
    # Update the data
    geosource.geojson = new_data

# Create a plotting function
def make_plot(month,day,urgency):    

  date=f"month:{month}:-day:{day}"
  # Set the format of the colorbar
  min_range = np.array(hex_grid_A1.filter(like=f"month:{month}:-day:{day}").min())[0]
  max_range = np.array(hex_grid_A1.filter(like=f"month:{month}:-day:{day}").max())[0]
  
  # Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
  color_mapper = LinearColorMapper(palette = palette, low = min_range, high = max_range)

  # Create color bar.
  color_bar = ColorBar(color_mapper=color_mapper, label_standoff=18,border_line_color=None, location = (0, 0))

  # Create figure object.
  p = figure(title = f'Average number of A{urgency} incidents on weekday {day} of Month {month}', 
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
p = make_plot(1,1,2)

# Make a MONTHS slider object 
mon_slider = Slider(title = 'Month',start = 1, end = test_months, step = 1, value = 1)
mon_slider.on_change('value', update_plot)
# Make a WEEKDAYS slider object
day_slider = Slider(title = 'Weekday',start = 1, end = test_days, step = 1, value = 1)
day_slider.on_change('value', update_plot)
# Make a URGENCY slider object 
urg_slider = Slider(title = 'Urgency A',start = 1, end = 2, step = 1, value = 2)
urg_slider.on_change('value', update_plot)

# Make a column layout of widgetbox(slider) and plot, and add it to the current document
# Display the current document
layout = column(p, widgetbox(mon_slider), widgetbox(day_slider), widgetbox(urg_slider))
curdoc().add_root(layout)
