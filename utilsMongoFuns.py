from bokeh.models import ColumnDataSource, FactorRange
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import widgetbox, row, column, gridplot, layout
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10, Category10_3
import pymongo
from pymongo import MongoClient
import pprint
import numpy as np
import pandas as pd
import datetime 
from itertools import compress
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import math
import panel
import sys
import time
#import hvplot.pandas # Adds .hvplot and .interactive methods to Pandas dataframes
import panel as pn # Panel is a simple, flexible and enterprise-ready data app framework
import numpy.ma as ma
import panel.widgets as pnw
from tqdm import tqdm
import seaborn as sns
import pickle
# import station_coordinates
pd.set_option('display.max_columns', None)

loc_to_sta_oost = {(52.335954653064526, 6.639791503190925): 'ALC', 
                    (52.22424359354638, 6.941124117194153): 'ENO', 
                    (52.2193571118841, 6.88415488625391): 'ENS', 
                    (52.239300040215454, 6.850546485451197): 'ENU', 
                    (52.3803817652154, 6.80648107037796): 'FLR', 
                    (52.285088464419985, 6.853713407669399): 'FodB', 
                    (52.2511458956598, 6.59602652699191): 'GOR', 
                    (52.28444902172763, 6.778520272671243): 'HEN', 
                    (52.1708937, 6.7413865): 'HKB', 
                    (52.1998457245631, 6.64497109895514): 'HNV', 
                    (52.2600872834731, 6.46119800195306): 'MRK', 
                    (52.37415307701571, 6.470170758530611): 'NVD', 
                    (52.3221934458346, 6.92607050386671): 'OLD', 
                    (52.3106786080303, 6.51759050316565): 'RSN',
                    (52.35366108006881, 6.586689405692712): 'SCP', 
                    (52.4052468135482, 6.78742225831179): 'TUB', 
                    (52.45268694795758, 6.577865358616354): 'VRM', 
                    (52.40562629264381, 6.63395867281057): 'VZV', 
                    (52.35155, 6.85717): 'WRS'}
stationsDict = {station:index for index,station in enumerate(loc_to_sta_oost.values())}
#define lists for weekdays, weeks, months
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
#round up, needed so that we can match coordinates of performed relocations to these stations. If we rounc up to too many digits, no matches will be found.

def adjust_time(timestr):
    if isinstance(timestr, float):  # In case there's a float
        return pd.NaT
    dt = datetime.datetime.strptime("".join(timestr[:10] + timestr[11:19]), '%Y-%m-%d%H:%M:%S')
    # Check if the datetime is on or after March 31, 2024, 1:00 AM
    if dt >= datetime.datetime(2024, 3, 31, 1, 0, 0):
        return dt + datetime.timedelta(hours=2)  # Add 2 hours
    else:
        return dt + datetime.timedelta(hours=1)  # Add 1 hour

def TimeUntilArrival012(dataset,start_date,end_date,regionId = 5): #5 for Twente
    """ Function that obtains total driving time in total_time_until_arrival column, as well as unique ids. (this function contains no filtering, so no ids that do not contain open column are filtered out.)
    Note that a problem is that the atRequest==True statement is in some cases (approx. 10-15 percent) not found, so that we cannot derive
    (at least not from that) the time the vehicle arrived, leading to some None values at the total_driving_time column. Need to
    obtain the finishing time from some other column. Look into this."""
    Assigned_query = {"$and": 
                    [{"requestUpdate.urgency": {"$in": ["A1", "A2", "A0"]}},
                    {"timeWrittenByLogger": {"$gte": start_date.isoformat() + 'Z',"$lte": end_date.isoformat() + 'Z'}},
                    {"requestUpdate.regionId": regionId},
                    {"requestUpdate.isRelocation":{"$exists":False}},
                    ]}
    assignedCars012 = pd.DataFrame(list(dataset['updates'].find(Assigned_query))) 
    assignedCars012['timestamp'] = assignedCars012['time'].apply(adjust_time)
    assignedCars012['timestampLogger'] = assignedCars012['timeWrittenByLogger'].apply(adjust_time)
    unique_keys_update = set()
    for d in assignedCars012.loc[pd.isnull(assignedCars012.requestUpdate)==False]['requestUpdate']:
        unique_keys_update.update(d.keys())
    for key in unique_keys_update:
        assignedCars012[key] = assignedCars012['requestUpdate'].apply(lambda row: row.get(key) if pd.isna(row)==False else np.nan)
    
    assignedCars012 = assignedCars012.reset_index().drop(['index','time','timeWrittenByLogger','_id','requestUpdate','regionId'],axis=1)

    assignedCars012 = assignedCars012.explode('dispatches').reset_index().drop('index',axis=1)
    unique_keys_disp = set()
    for d in assignedCars012.loc[pd.isnull(assignedCars012.dispatches)==False]['dispatches']:
        unique_keys_disp.update(d.keys())

    for key in unique_keys_disp:
        assignedCars012[key] = assignedCars012['dispatches'].apply(lambda row: row.get(key) if pd.isna(row)==False else np.nan)

    assignedCars012['requestId'] = assignedCars012['requestId'].astype(str)
    #now, make sure to convert all columns to objects (string type), except for the datetime columns
    columns_not_objects = ['timestamp','timestampLogger']
    assignedCars012 = assignedCars012.apply(lambda col: col if col.name in columns_not_objects else col.astype(str))
    unique_ids = np.unique(assignedCars012.requestId)
    assignedCars012 = assignedCars012.drop('dispatches',axis=1).dropna(axis=1, how='all').reset_index().drop('index',axis=1).sort_index(axis=1)
    #time_gap describes inbetween timestamp and timestampLogger
    assignedCars012['time_gap'] =(assignedCars012.timestamp-assignedCars012.timestampLogger).dt.total_seconds()
    # Calculating the start and end times and time differences
    start_times = assignedCars012[assignedCars012['open']==True].groupby('requestId')['timestamp'].first().rename('start_time')
    end_times = assignedCars012[assignedCars012['atRequest']==True].groupby('requestId')['timestamp'].first().rename('end_time')
    times_df = pd.merge(start_times, end_times, on='requestId')
    times_df['total_time_until_arrival'] = (times_df['end_time'] - times_df['start_time']).dt.total_seconds()
    assignedCars012 = pd.merge(assignedCars012, times_df['total_time_until_arrival'], on='requestId', how='left')    
    return assignedCars012,unique_ids

def FindAssignedCars012(dataset,start_date,end_date,regionId = 5):
    """  Function that extracts which cars were assigned to an incident in region 5 from updates Collection. So a filtered version of TimeUntilArrival012.
    But on the other hand no atRequest==True filters. """
    Assigned_query = {"$and": 
                    [{"requestUpdate.urgency": {"$in": ["A1", "A2", "A0"]}},
                    {"timeWrittenByLogger": {"$gte": start_date.isoformat() + 'Z',"$lte": end_date.isoformat() + 'Z'}},
                    {"requestUpdate.regionId": regionId},
                    {"requestUpdate.isRelocation":{"$exists":False}},
                    {"requestUpdate.dispatches.coupledVehicle": {"$exists":True}} #checks whether a vehicle is coupled to an incident. This factor should be present for checking when a car is assigned to an incident
                    ]}
    assignedCars012 = pd.DataFrame(list(dataset['updates'].find(Assigned_query))) 
    assignedCars012['timestamp'] = assignedCars012['time'].apply(adjust_time)
    assignedCars012['timestampLogger'] = assignedCars012['timeWrittenByLogger'].apply(adjust_time)
    unique_keys_update = set()
    for d in assignedCars012.loc[pd.isnull(assignedCars012.requestUpdate)==False]['requestUpdate']:
        unique_keys_update.update(d.keys())
    for key in unique_keys_update:
        assignedCars012[key] = assignedCars012['requestUpdate'].apply(lambda row: row.get(key) if pd.isna(row)==False else np.nan) 

    assignedCars012 = assignedCars012.reset_index().drop(['index','time','timeWrittenByLogger','_id','requestUpdate','regionId'],axis=1)
    assignedCars012 = assignedCars012.explode('dispatches').reset_index().drop('index',axis=1)
    unique_keys_disp = set()
    for d in assignedCars012.loc[pd.isnull(assignedCars012.dispatches)==False]['dispatches']:
        unique_keys_disp.update(d.keys())
    for key in unique_keys_disp:
        assignedCars012[key] = assignedCars012['dispatches'].apply(lambda row: row.get(key) if pd.isna(row)==False else np.nan)

    columns_not_objects = ['timestamp','timestampLogger']
    assignedCars012 = assignedCars012.apply(lambda col: col if col.name in columns_not_objects else col.astype(str))
    assignedCars012 = assignedCars012.drop('dispatches',axis=1).dropna(axis=1, how='all').reset_index().drop('index',axis=1).sort_index(axis=1)
    assignedCars012['time_gap'] =(assignedCars012.timestamp-assignedCars012.timestampLogger).dt.total_seconds()
    assignedCars012 = assignedCars012[np.invert(assignedCars012.duplicated(subset=['requestId','coupledVehicle']))].reset_index().drop(['index'],axis=1)

    return assignedCars012

def FindDispatchAdvices(area,previous,dataset,start_date,end_date,unique_ids):
    start_date_string = start_date.strftime("%Y_%m_%d")
    end_date_string = end_date.strftime("%Y_%m_%d")

    """ With this function we find the non-na dispatch advices, as well as the unique ones and the #advs per incident.
    We also create a plot on the number of advices and unique advices. Should take about one hour per simulated month"""

    if previous and os.path.exists(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_dispatchAdvices"):
        with open(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_dispatchAdvices", 'rb') as file:
            print("Use pickle results.")
            dispatchAdvices = pickle.load(file)
            
    else:
        dispadv_query = {"$and": 
                        [{"timeWrittenByLogger": {"$gte": start_date.isoformat() + 'Z',"$lte": end_date.isoformat() + 'Z'}},
                        {"advices": {"$exists":True}}
                        ]}
        dispatchAdvices = pd.DataFrame(list(dataset.dispatchAdvices.find(dispadv_query)))
        print('dataframe obtained.')
        dispatchAdvices['timestamp'] = dispatchAdvices['time'].apply(adjust_time)
        dispatchAdvices['timestampLogger'] = dispatchAdvices['timeWrittenByLogger'].apply(adjust_time)
        
        dispatchAdvices = dispatchAdvices.explode('advices',ignore_index=True)
        dispatchAdvices.drop(['_id','time','timeWrittenByLogger'],axis=1,inplace=True)
        unique_keys_adv = set()
        for d in dispatchAdvices['advices']:
            unique_keys_adv.update(d.keys())

        for key in tqdm(unique_keys_adv):
            dispatchAdvices[key] = dispatchAdvices['advices'].apply(lambda row: row.get(key) if pd.isna(row)==False else np.nan)
        dispatchAdvices.drop(['advices'], axis=1, inplace=True)
        # Downcast each float column to float32 to save some memory
        float_columns = ['coverageAfterDispatch', 'coverageDifferenceAfterDispatch', 'drivingTime', 'coupledVehicleDrivingTime']
        for col in float_columns:
            if col in dispatchAdvices.columns:
                dispatchAdvices[col] = pd.to_numeric(dispatchAdvices[col], downcast='float')

        dispatchAdvices['requestId'] = dispatchAdvices['requestId'].astype(str)
        # dispatchAdvices.sort_values(by=['timestamp'], inplace=True)
        dispatchAdvices.rename(columns={'vehicleId': 'vehicleCode'}, inplace=True)
        dispatchAdvices = dispatchAdvices[dispatchAdvices['requestId'].isin(unique_ids)].reset_index(drop=True)

        # dispatchAdvices = dispatchAdvices.sort_values(by=['timestamp']).reset_index(drop=True).drop(['advices'],axis=1)
        # dispatchAdvices = dispatchAdvices.rename(columns={'vehicleId' : 'vehicleCode'})
        # dispatchAdvices = dispatchAdvices[dispatchAdvices['requestId'].isin(unique_ids)].reset_index(drop=True) #only keep requestIDs that are present in updates dataset.
        # #add a column that describes whether dispatch advice was first / second / ... best option out of all options at that timestamp
        # #note, NA entries for advice_rank are because drivingTime is NA for this entry.

        dispatchAdvices['advice_rank'] = dispatchAdvices.groupby(['timestamp','requestId'])['drivingTime'].rank(method='dense', ascending=True)
        #add potential end timestamps to the dispatch_advices (dispAdvices) dataframe
        dispatchAdvices['potential_end_time'] = dispatchAdvices['timestamp'] + pd.to_timedelta(dispatchAdvices['drivingTime'], unit='s')
        with open(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_dispatchAdvices", 'wb') as file:
            pickle.dump(dispatchAdvices, file)

    #unique advices.
    if previous and os.path.exists(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_uniquedispatchAdvices"):
        with open(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_uniquedispatchAdvices", 'rb') as file:
            print("Use pickle results for unique dispatch advices as well.")
            uniquedispatchAdvices = pickle.load(file)
    else:
        uniquedispatchAdvices = dispatchAdvices.drop_duplicates(subset=['requestId', 'vehicleCode']).reset_index(drop=True)
        with open(f"regions/{area}/saved_results/{start_date_string}_{end_date_string}_uniquedispatchAdvices", 'wb') as file:
            pickle.dump(uniquedispatchAdvices, file)


    #advices per incident
    uniqueAdvsPerIncident = list(uniquedispatchAdvices.groupby(['requestId']).count().timestamp)
    advicesPerIncident= list(dispatchAdvices.groupby(['requestId']).count().timestamp)
    advicesPerIncident.remove(max(advicesPerIncident)) #some reqID can get tens of thousands of advices

    print(f"The average (mean) number of advices per incident is {int(np.mean(advicesPerIncident))} while the median number of advices is {int(np.median(advicesPerIncident))}.")
    print(f"The average (mean) number of unique advices per incident is {round(np.mean(uniqueAdvsPerIncident),2)} while the median number of unique advices is {round(np.median(uniqueAdvsPerIncident))}.")
    print(f"Between {start_date.strftime('%m/%d/%Y')} and {end_date.strftime('%m/%d/%Y')}, the number of "+
          f"unique dispatch advices for urgencies A0/A1/A2 is {len(uniquedispatchAdvices)}. The total number of dispatch advices for A-incidents is {len(dispatchAdvices)}.")
    print(f"The average (mean) number of unique daily advices is {round(len(uniquedispatchAdvices)/((end_date-start_date).days+1),2)}.")

    
    advicesPerIncident_dict = {'Total': advicesPerIncident, 'Unique': uniqueAdvsPerIncident}
    #make a plot about this.
    dict_perinc = advicesPerIncident_dict
    return dispatchAdvices,uniquedispatchAdvices,advicesPerIncident_dict

def CompareDispatch(assignedCars, dispAdvices):
    '''for given dataframes assignedCars (all assigned cars to incidents) and dispAdvices (all dispatch advices),
    this function creates statistics that show how many (and whether it was 1st / 2nd / ... option) advices were followed'''
    # Create a new dataframe based on assignedCars with selected columns
    df_new = assignedCars[['timestamp', 'requestId', 'coupledVehicle', 'urgency']].copy()
    print(type(assignedCars.requestId[0]),type(assignedCars.coupledVehicle[0]),type(dispAdvices.requestId[0]),type(dispAdvices.vehicleCode[0]))
    # Merge with dispAdvices to find matches, using the indicator=True to get the _merge column
    merged = pd.merge(
        df_new,
        dispAdvices[['requestId', 'vehicleCode']],
        left_on=['requestId', 'coupledVehicle'],
        right_on=['requestId', 'vehicleCode'],
        how='left',
        indicator=True
    )
    # Create the adviceGiven column based on the presence of matches
    df_new['adviceGiven'] = merged['_merge'] == 'both'
    # Merge to get the closest timestamps and rename potential_end_time to actual_end_time
    df_new = pd.merge_asof(
        df_new.sort_values('timestamp'),
        dispAdvices.sort_values('timestamp'),
        left_on='timestamp',
        right_on='timestamp',
        left_by=['requestId', 'coupledVehicle'],
        right_by=['requestId', 'vehicleCode'],
        direction='backward'
    ).rename(columns={'potential_end_time': 'actual_end_time'})
    
    # Compute the optimal end time for each requestId and merge with the main dataframe to add the optimal_end_time
    optimal_times = dispAdvices.groupby('requestId')['potential_end_time'].min().reset_index()
    optimal_times.rename(columns={'potential_end_time': 'optimal_end_time'}, inplace=True)
    
    df_final = pd.merge(df_new, optimal_times, on='requestId', how='left')
    df_final = df_final[['timestamp', 'requestId', 'coupledVehicle', 'adviceGiven', 'advice_rank', 'actual_end_time', 'optimal_end_time','urgency']]
    
    # Convert timestamp to datetime and extract week number and day
    df_final['day'] = df_final['timestamp'].dt.day
    df_final['weekday'] = df_final['timestamp'].dt.weekday
    # df_final['week'] = df_final['timestamp'].dt.to_period('W').apply(lambda r: r.start_time.strftime('%Y-%m-%d'))
    # df_final['month'] = df_final['timestamp'].dt.to_period('M').apply(lambda r: r.start_time.strftime('%Y-%m'))
    df_final['week'] = df_final['timestamp'].dt.isocalendar().week
    df_final['month'] = df_final['timestamp'].dt.month

    return df_final

def DispatchRanksNumsPlots(DispStats_df, group_by_period='week'):
    ''' Make ranking with the dispatch statistics dataframe group_by_period should be either day, week, or month'''
    
    DispStats_df['time_difference'] = (DispStats_df['actual_end_time'] - DispStats_df['optimal_end_time']).dt.total_seconds()
    # Filter out rows with None in advice_rank and calculate percentages
    df_filtered = DispStats_df.copy()
    df_filtered = df_filtered.dropna(subset=['advice_rank'])
    df_filtered['advice_rank'] = df_filtered['advice_rank'].astype(int)

    # Group advice ranks of 5 or higher into one category
    df_filtered['advice_rank_grouped'] = df_filtered['advice_rank'].apply(lambda x: x if x <= 4 else 5)
    advice_rank_counts = df_filtered.groupby([group_by_period, 'advice_rank_grouped']).size().unstack(fill_value=0)
    advice_rank_percentages = advice_rank_counts.div(advice_rank_counts.sum(axis=1), axis=0) * 100

    # Plot the percentages of each advice_rank
    DispStats_df['time_difference'] = (DispStats_df['actual_end_time'] - DispStats_df['optimal_end_time']).dt.total_seconds()
    # Filter out rows with None in advice_rank and calculate percentages
    df_filtered = DispStats_df.copy()
    df_filtered = df_filtered.dropna(subset=['advice_rank'])
    df_filtered['advice_rank'] = df_filtered['advice_rank'].astype(int)

    # Group advice ranks of 5 or higher into one category
    df_filtered['advice_rank_grouped'] = df_filtered['advice_rank'].apply(lambda x: x if x <= 4 else 5)
    advice_rank_counts = df_filtered.groupby([group_by_period, 'advice_rank_grouped']).size().unstack(fill_value=0)
    advice_rank_percentages = advice_rank_counts.div(advice_rank_counts.sum(axis=1), axis=0) * 100

    df = advice_rank_percentages
    weeks        = df.index.to_list()
    df_weeks_str = [str(week) for week in weeks]
    data_dict = {'weeks' : df_weeks_str}
    for rank in df.columns.to_list():
        data_dict[f"{rank}"] = df[rank].values
    advice_ranks = df.columns.to_list()  # Ensure these match your actual column names

    # Now prepare the incidents vs dispatches plot
    incidents_per_period = DispStats_df.groupby(group_by_period)['requestId'].nunique()
    dispatches_per_period = DispStats_df.groupby(group_by_period).size()
    comparison_df = pd.DataFrame({
            '# Incidents': incidents_per_period,
            '# Dispatches': dispatches_per_period }).reset_index()
    incVdisp         = [ (str(week), column) for week in comparison_df.week for column in comparison_df.columns[1:]]
    incVdisp_counts  = comparison_df[comparison_df.columns[1:]].values.flatten().tolist()
    incVdisp_source  = ColumnDataSource(data=dict(x=incVdisp, counts=incVdisp_counts))

    palette = Category10_3  # You can also choose another palette with more colors if needed
    colors = Category10[len(advice_ranks)]  # Use a color palette with enough colors for each rank

    p2 = figure(x_range=FactorRange(*incVdisp), height=500, 
                title=f'Number of Incidents vs Number of Dispatches per Week',
                toolbar_location=None, tools="")

    p2.vbar(x='x', top='counts', width=0.9, source=incVdisp_source, line_color="white", 
            fill_color=factor_cmap('x', palette=palette, factors=['# Incidents', '# Dispatches'], start=1, end=2))

    # Customize the plot appearance
    p2.y_range.start = 0
    p2.x_range.range_padding = 0.1
    p2.xgrid.grid_line_color = None

    p1 = figure(x_range=df_weeks_str, plot_height=600, 
            title   = f'Incident Advice Distribution per {group_by_period.capitalize()}', 
            toolbar_location=None, tools="", y_range=(0, 100))

    # Plot the stacked bars
    p1.vbar_stack([str(rank) for rank in advice_ranks],x='weeks', width=0.9, color=colors, source=data_dict, 
                legend_label=[f'Advice {rank}' if int(rank) <= 4 else 'Advice 5 - Higher' for rank in advice_ranks])
    # Customize plot labels
    p1.xaxis.axis_label = group_by_period.capitalize()  # Example: 'Week'
    p1.yaxis.axis_label = 'Percentage of Incidents'

    # Set y-axis ticks and labels
    p1.yaxis.ticker = [i for i in range(0, 101, 10)]
    p1.yaxis.major_label_overrides = {i: f'{i}%' for i in range(0, 101, 10)}
    # Customize legend
    p1.legend.title = 'Advice Ranks'
    p1.legend.location = 'left'
    p1.legend.orientation = 'horizontal'

    return row(p1,p2)

def mongoDBimportTwente(area,startMonth,startDay,endMonth,endDay,boolean):
    loc_to_sta_rounded = {tuple(np.round(key,2)) : value for key, value in loc_to_sta_oost.items()}
    previous = [True,False][boolean]
    client = MongoClient("mongodb+srv://seconds:test%5EMe%5E%5E@cluster0.z9k9jkv.mongodb.net/")
    dataset = client.aon_prd_V2
    start_date = datetime.datetime(2024, startMonth, startDay, 0 , 0, 0)
    #since 2 hours are added after 31 of March, in order to get data until end of march, do until 22:00
    end_date = datetime.datetime(2024, endMonth, endDay, 23, 59, 59)

    TimeUntilArrival012_df, uniqueIDs                             = TimeUntilArrival012(dataset,start_date,end_date)
    print('TimeUntilArrival done')
    AssignedCars_df                                               = FindAssignedCars012(dataset,start_date,end_date)
    print('Assigned Cars done')
    DispAdvices_df, UniqueDispAdvices_df, AdvicesPerIncident_dict = FindDispatchAdvices(area,previous,dataset,start_date,end_date,uniqueIDs)
    print('DispatchAdvices done')
    DispStats_df                                                  = CompareDispatch(AssignedCars_df,DispAdvices_df)
    print('DispatchStats done')

    return DispatchRanksNumsPlots(DispStats_df)

# mongoDBimportTwente(8,1,8,5,1)