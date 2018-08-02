# This tool is for tagging data, visualizations
# Future uses:  Visualization
#               Comparison of different algorithms
# Developer:    Rohit Aggarwal
# Dated:        July 20, 2018
# All hardcoded parameters have a comment # HARDCODED

import dash
from collections import defaultdict
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_table_experiments as dtable
from dateutil.relativedelta import relativedelta
import pandas as pd
import flask
from flask_cors import CORS
import os
import datetime as dt
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import base64
import urllib
import json
import functools
from flask_caching import Cache
from textwrap import dedent
import sys
from api import base

app = dash.Dash('GTtagger')
filename = sys.argv[-1]

styles = {
    'pre':  {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
}
}

server = app.server

class keydefaultdict(defaultdict):
    """Subclasss defaultdict such that the default_factory method called for missing
        keys takes a single parameter https://stackoverflow.com/a/2912455/77533
        """
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret

FILE_MAP = {}
UUID_list = pd.read_csv(filename)
for i in UUID_list['uuid']:
    FILE_MAP[i] = i

@functools.lru_cache(maxsize=32000)
def getData(uuid):
    
    output = {}
    
    cache = os.path.dirname(os.path.abspath(__file__)) + '/'
    input_fileName = cache + 'GTtagger/'+uuid+'.csv'
    output_fileName = cache + 'GTtagger_output/'+uuid+'.csv'
    vacation_input_fileName = cache + 'Vacation_input/'+uuid+'_preddates.csv'
    timezone = 'UTC'

    if os.path.isfile(input_fileName):
        df=pd.read_csv(input_fileName)
    elif os.path.isfile(cache+'GTtagger/'+uuid+'.csv.gz'):
        df=pd.read_csv(cache+'GTtagger/'+uuid+'.csv.gz', compression='gzip')
    df['Fractional_hour'] = df['Hour'] + (df.Timestamp % 3600)/3600.0
    nSamplesPerHour = 3600/(df.Timestamp.diff().median())
    nSamplesPerHour = int(nSamplesPerHour)

    dailyMinConsumption = pd.DataFrame(df.groupby(['Date'])['Value'].min())
    dailyMinConsumption = dailyMinConsumption.reset_index(drop=False)
    dailyMinConsumption = dailyMinConsumption.fillna(0)
    dailyMinConsumption = dailyMinConsumption.rename(columns = {'Value':'dailyMinCons'})
    df = pd.merge(df, dailyMinConsumption, on=['Date'])

    dailyConsumption = pd.DataFrame(df.groupby(['Date'])['Value'].sum()/(df.groupby(['Date'])['Value'].count().max()*1.0)*24.0/1000.0)
    dailyConsumption = dailyConsumption.reset_index(drop=False)
    dailyConsumption = dailyConsumption.fillna(0)

    dailyMaxConsumption = pd.DataFrame(df.groupby(['Date'])['Value'].max())
    dailyMaxConsumption = dailyMaxConsumption.reset_index(drop=False)
    dailyMaxConsumption = dailyMaxConsumption.fillna(0)

    maxDailyEnergy = np.max(dailyConsumption['Value']).astype(int)
    minDailyEnergy = np.min(np.min(dailyConsumption['Value']).astype(int), 0)

    ymax = np.max(df['Value'])*nSamplesPerHour
    ymin = np.min(np.min(df['Value'])*nSamplesPerHour,0)
    maxenergy = np.ceil(np.max(df['Value'])/500)*500
    maxenergy = maxenergy.astype(int)
    minenergy = np.min(np.floor(np.min(df['Value'])/500)*500, 0)
    minenergy = minenergy.astype(int)

    sample_rate = df.Timestamp.diff().median()
    df_local = df[~df.Date.isnull()].copy()
    df_local['Hour2'] = df_local['Hour'] + (df_local.Timestamp % 3600)/3600.0

    dates, idx_dates = np.unique(df_local.Date, return_inverse=True)
    hours, idx_hours = np.unique(df_local.Hour2, return_inverse=True)

    date_dt = pd.to_datetime(dates, format='%Y-%m-%d')
    original_datelist = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    A = np.zeros((dates.shape[0], hours.shape[0]))
    A[idx_dates, idx_hours] = df_local.Value
    
    months, idx_months = np.unique(pd.to_datetime(dates).strftime('%Y-%m'), return_index=True)

    STARTING_DATE = dates[0]
    DAILY_IMG = df.loc[df['Date'] == STARTING_DATE][['Fractional_hour','Hour','Value']]

    if os.path.isfile(vacation_input_fileName):
        vacation_data = pd.read_csv(vacation_input_fileName, header = 0)
        vac_data = [None]*len(vacation_data['Timestamp']);
        count = 0
        for i in vacation_data['Timestamp']:
            vac_data[count] = str(base.get_local_datetime(i, timezone).strftime('%Y-%m-%d'))
            count = count + 1
        output['vacation_dates'] = vac_data
    else:
        output['vacation_dates'] = dailyConsumption[dailyConsumption.Value < np.percentile(dailyConsumption.Value, 30)]['Date']
        output['vacation_dates'] = list(output['vacation_dates'].values)

    output['df'] = df
    output['vacation_dates_backup'] = output['vacation_dates']
    output['output_fileName'] = output_fileName
    output['DAILY_IMG'] = DAILY_IMG
    output['STARTING_DATE'] = STARTING_DATE
    output['A'] = A
    output['date_dt'] = date_dt
    output['original_datelist'] = original_datelist
    output['sample_rate'] = sample_rate
    output['minenergy'] = minenergy
    output['maxenergy'] = maxenergy
    output['ymin'] = ymin
    output['ymax'] = ymax
    output['dates'] = dates
    output['hours'] = hours
    output['minDailyEnergy'] = minDailyEnergy
    output['maxDailyEnergy'] = maxDailyEnergy
    output['dailyConsumption'] = dailyConsumption
    output['dailyMaxConsumption'] = dailyMaxConsumption
    output['nSamplesPerHour'] = nSamplesPerHour
    output['click_data'] = None
    return output

DATA_FRAMES = keydefaultdict(lambda name: getData((FILE_MAP[name])))
dataF = {}
for (count, i) in enumerate(FILE_MAP):
    print('Loading ....', i)
    dataF[i] = DATA_FRAMES[i]
    if count == 0:
        df = dataF[i]['df']
        vacation_dates = dataF[i]['vacation_dates']
        DAILY_IMG = dataF[i]['DAILY_IMG']
        STARTING_DATE = dataF[i]['STARTING_DATE']
        A = dataF[i]['A']
        date_dt = dataF[i]['date_dt']
        original_datelist = dataF[i]['original_datelist']
        sample_rate = dataF[i]['sample_rate']
        minenergy = dataF[i]['minenergy']
        maxenergy = dataF[i]['maxenergy']
        ymin = dataF[i]['ymin']
        ymax = dataF[i]['ymax']
        dates = dataF[i]['dates']
        hours = dataF[i]['hours']
        minDailyEnergy = dataF[i]['minDailyEnergy']
        maxDailyEnergy = dataF[i]['maxDailyEnergy']
        dailyConsumption = dataF[i]['dailyConsumption']
        nSamplesPerHour = dataF[i]['nSamplesPerHour']

def generate_table(dataframe, max_rows=10):
    return html.Table(
                      # Header
                      [html.Tr([html.Th(col) for col in dataframe.columns])] +
                      
                      # Body
                      [html.Tr([
                                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                                ]) for i in range(min(len(dataframe), max_rows))]
                      )

app.layout = html.Div([
                        # Row 1: Header and Intro text
                        html.Div([
                        html.Div([
                            html.H2('GT tagger',
                                    style={
                                        'position': 'relative',
                                        'top': '0px',
                                        'left': '10px',
                                        'font-family': 'Dosis',
                                        'display': 'inline',
                                        'font-size': '6.0rem',
                                        'color': '#4D637F'
                                    }),
                            html.H2('for',
                                    style={
                                        'position': 'relative',
                                        'top': '0px',
                                        'left': '20px',
                                        'font-family': 'Dosis',
                                        'display': 'inline',
                                        'font-size': '4.0rem',
                                        'color': '#4D637F'
                                    }),
                            html.H2('Vacation/PP/HVAC',
                                    style={
                                        'position': 'relative',
                                        'top': '0px',
                                        'left': '27px',
                                        'font-family': 'Dosis',
                                        'display': 'inline',
                                        'font-size': '6.0rem',
                                        'color': '#4D637F'
                                    }),
                        ], className='three rows', style={'position': 'relative', 'right': '15px', 'margin-left': '20px'}),
                                html.Div([
                                    html.Div([
                                        html.P('HOVER over a day in the graph to the right to see its plot to the left.'),
                                    ], style={'margin-left': '20px'}),
                                          
                                          html.Button('Reset Filters', id='resetFilter', style={'position': 'relative', 'margin-left': '35px'}),
                                          html.Div(id='output'),
                                          ], className='two rows' ),
                                  ], className='row' ),
                       html.Div([
                                  html.Label('UUID selector'),
                                  dcc.Dropdown(id = 'dropdown',
                                               options=[{'label': i, 'value': i} for i in FILE_MAP.keys()],
                                               value=FILE_MAP[list(FILE_MAP.keys())[0]],
                                               ),
                                 html.Hr(),
                                 ], className='row', style = {'margin-left': '35px', 'width': '1000px'} ),
                       html.Div([
                                 html.Div([
                                           html.Label('Hour filter'),
                                           dcc.RangeSlider(
                                                           id='hourFilter',
                                                           marks={i: i/2 for i in range(0, 24*nSamplesPerHour-1, 2)},
                                                           min=0,
                                                           max=24*nSamplesPerHour,
                                                           step = 1,
                                                           value=[0, 24*nSamplesPerHour-1]
                                                           ),
                                           html.Hr(),
                                           ],className='row', style = {'margin-left': '35px', 'width': '1000px'}),
                                 html.Div([
                                           html.Label('Amp filter (WH)'),
                                           dcc.RangeSlider(
                                                           id='ampFilter',
                                                           marks={i: i for i in range(minenergy, maxenergy, int(int(np.round((maxenergy-minenergy)/10.0)/100.0)*100.0))},
                                                           min=minenergy,
                                                           max=maxenergy,
                                                           step = 100,
                                                           value=[minenergy, maxenergy]
                                                           ),
                                           html.Hr(),
                                           ],className='row', style = {'margin-left': '35px', 'width': '1000px'}),
                                 html.Div([
                                           html.Label('Daily maxAmp filter (WH)'),
                                           dcc.RangeSlider(
                                                           id='maxAmpFilter',
                                                           marks={i: i for i in range(minenergy, maxenergy, int(int(np.round((maxenergy-minenergy)/10.0)/100.0)*100.0))},
                                                           min=minenergy,
                                                           max=maxenergy,
                                                           step = 100,
                                                           value=[minenergy, maxenergy]
                                                           ),
                                           html.Hr(),
                                           ],className='row', style = {'margin-left': '35px', 'width': '1000px'}),
                                 html.Div([
                                           html.Label('Date filter'),
                                           dcc.RangeSlider(
                                                           id='dateFilter',
                                                           marks={i: (str(date_dt[i].year)+'-'+str(date_dt[i].month)) for i in range(0,len(date_dt),int(len(date_dt)/6))},
                                                           min=0,
                                                           max=len(date_dt)-1,
#                                                           step = 15,
                                                           value=[0, len(date_dt)-1],
                                                           ),
                                           html.Hr(),
                                           ],className='row', style = {'margin-left': '35px', 'width': '1000px'}),
                                 html.Div([
                                           html.Label('dailyCons filter (KwH)'),
                                           dcc.RangeSlider(
                                                           id='dailyConsFilter',
                                                           marks={i: i for i in range(minDailyEnergy, maxDailyEnergy,  int(max((maxDailyEnergy-minDailyEnergy)/10.0,1)))},
                                                           min=minDailyEnergy,
                                                           max=maxDailyEnergy,
                                                           step = 0.1,
                                                           value=[minDailyEnergy, maxDailyEnergy]
                                                           ),
                                           html.Hr(),
                                           ],className='row', style = {'margin-left': '35px', 'width': '1000px'}),
                                   html.Div([
                                             html.Div([
                                                        dcc.Graph(
                                                                  id = "dailyPlot",
                                                                  style=dict(width='700px'),
                                                                  figure = go.Figure(
                                                                                     data = [go.Scatter(
                                                                                                        y = DAILY_IMG['Fractional_hour'],
                                                                                                        x = DAILY_IMG['Value']*nSamplesPerHour,
                                                                                                        mode = 'lines+markers',
                                                                                                        )],
                                                                                     layout = go.Layout(
                                                                                                        title='Daily Chart',
                                                                                                        height = 400,
                                                                                                        xaxis = dict(title='Hours', nticks=12, range = [0,23],),
                                                                                                        yaxis = dict(title='Power (W)', ticks='', range = [ymin, ymax] )
                                                                                                        ))
                                                                  ),
                                                       dcc.Graph(id='histogram-graph',
                                                                 style=dict(width='700px'),
                                                                 figure=go.Figure(
                                                                                  data = [go.Histogram(
                                                                                                       x = dailyConsumption['Value']
                                                                                                       )],
                                                                                  layout = go.Layout(
                                                                                                     title='Heatmap of daily consumption (KwH)',
                                                                                                     height = 400,
                                                                                                     xaxis = dict(title='Hours', ticks=''),
                                                                                                     yaxis = dict(title='# occurrences', ticks='' ))) )
                                                       ], className='two rows', style = {'margin-left': '-125px'}),
                                            ], className='two columns'),
                                   html.Div([
                                             dcc.Graph(id='clickable-graph',
                                                       style=dict(width='700px'),
                                                       hoverData=dict( points=[dict(pointNumber=0)] ),
                                                       figure=go.Figure(
                                                                        data = [go.Heatmap(
                                                                                           z=A,
                                                                                           x=hours,
                                                                                           y=dates,
                                                                                           colorscale='Jet',
                                                                                           zsmooth= False,
                                                                                           )],
                                                                        layout = go.Layout(
                                                                                           title='Heatmap',
                                                                                           height = 800,
                                                                                           xaxis = dict(ticks=''),
                                                                                           yaxis = dict(ticks='' ))) ),
                                             ],className='two columns', style={'position': 'relative', 'margin-left': '15px'}),
                                 ],className='row', style=dict(width='4000px', height='1000px') ),
                       html.Hr(),
                       html.Hr(),
                       html.Hr(),
                       html.Hr(),
                       html.Hr(),
                       html.Hr(),
                       html.Div([
                                 html.Label('Options'),
                                 dcc.Dropdown(id = 'options',
                                              options=[{'label': 'all', 'value': 'all'}, {'label': 'vacation', 'value': 'vacation'}, {'label': 'non-vacation', 'value': 'non-vacation'}],
                                              value='all',
                                              ),
                                 html.Hr(),
                                 ], className='row', style = {'margin-left': '35px', 'width': '500px'} ),
                       html.Div([
                                 html.Label('Edit mode'),
                                 dcc.Dropdown(id = 'vac_togggler',
                                              options=[{'label': 'Enable vacation/non-vacation toggle via click', 'value': 'enable_click'}, {'label': 'None', 'value': 'disable_click'}],
                                              value='disable_click',
                                              ),
                                 html.Hr(),
                                 ], className='row', style = {'margin-left': '35px', 'width': '500px'} ),
                       html.Div([dcc.RadioItems(
                                      id='toggle',
                                      options=[{'label': i, 'value': i} for i in ['New File', 'Append old file']],
                                      value='New File'
                                      ),
                                 ],className='two columns', style={'position': 'relative', 'margin-left': '35px'}),
#                       html.Button('Reset Vacation Data', id='reset-vac-data', style={'position': 'relative', 'margin-left': '35px'}),
                       html.Div([dcc.RadioItems(
                                      id='reset-vac-data',
                                      options=[
                                               {'label': 'Reset vacation data', 'value': True},
                                               {'label': 'Do nothing', 'value': False},
                                               ],
                                      value=False
                                      )],className='two columns',style={'position': 'relative', 'margin-left': '435px'}),
                       html.Hr(),
                       html.Hr(),
                       html.Button('Download File', id='download-data', style={'position': 'relative', 'margin-left': '35px'}),
                       html.Div(id='output-container-button',
                                children='Enter a value and press submit', style={'position': 'relative', 'margin-left': '35px'}),
                       ], className='row' )

@app.callback(
              Output('output-container-button', 'children'),
              [Input('download-data', 'n_clicks'),
               Input('reset-vac-data', 'value')],
               state=[
                      State('ampFilter', 'value'),
                      State('maxAmpFilter', 'value'),
                      State('hourFilter', 'value'),
                      State('dailyConsFilter', 'value'),
                      State('dateFilter', 'value'),
                      State('toggle', 'value'),
                      State('dropdown', 'value'),
                      State('options', 'value'),],
                    )
def updateTable(n_clicks, vacation_reset, ampValue, maxAmpValue, hourValue, dailyConsValue, dateRange, toggleValue, uuid, option_value):
    if vacation_reset:
        dataF[uuid]['vacation_dates'] = dataF[uuid]['vacation_dates_backup']
        return 'Vacation data restored to original. No file has been saved. Select option \"DO NOTHING\" to save data'
    dailyConsumption = dataF[uuid]['dailyConsumption']
    dailyMaxConsumption = dataF[uuid]['dailyMaxConsumption']
    df = dataF[uuid]['df']
    output_fileName = dataF[uuid]['output_fileName']
    original_datelist = dataF[uuid]['original_datelist']
    vacation_dates = dataF[uuid]['vacation_dates']
    dates = dataF[uuid]['dates']
    
    if n_clicks:
        validDates = dailyConsumption[(dailyConsumption['Value'] >= dailyConsValue[0]) & (dailyConsumption['Value'] <= dailyConsValue[1]) & (dailyMaxConsumption['Value'] >= maxAmpValue[0]) & (dailyMaxConsumption['Value'] <= maxAmpValue[1])].Date
        if (option_value == 'vacation'):
            vac_df = pd.DataFrame(vacation_dates, columns = ['Date'])
            index_test = validDates.isin(vac_df['Date'])
            validDates = validDates[index_test]
        elif (option_value == 'non-vacation'):
            vac_df = pd.DataFrame(vacation_dates, columns = ['Date'])
            index_test = validDates.isin(vac_df['Date'])
            validDates = validDates[~index_test]
        
        DAILY_IMG2 = df[((df['Hour'] >= hourValue[0]/2.0) & (df['Hour'] <= hourValue[1]/2.0) & (df['Value'] >= ampValue[0]) & (df['Value'] <= ampValue[1]) & (original_datelist.isin(validDates)) & (original_datelist >= date_dt[dateRange[0]]) & (original_datelist <= date_dt[dateRange[1]]))]
        if (os.path.isfile(output_fileName) & (toggleValue == 'Append old file')):
            DAILY_IMG2.to_csv(output_fileName, mode='a', index = False, header=False)
        else:
            DAILY_IMG2.to_csv(output_fileName, index = False)
        return 'The button has been clicked {} times. Number of records stored = {}. Filename = {}'.format(n_clicks, DAILY_IMG2.shape[0], output_fileName)
    else:
        return 'Button not clicked'

@app.callback(
              Output('dailyConsFilter', 'value'),
              [Input('resetFilter', 'n_clicks'),
               Input('dropdown', 'value'),]
              )
def resetFilterFunction(n_clicks, uuid):
    minDailyEnergy = dataF[uuid]['minDailyEnergy']
    maxDailyEnergy = dataF[uuid]['maxDailyEnergy']
    dailyConsValue = [minDailyEnergy, maxDailyEnergy]
    return dailyConsValue

@app.callback(
              Output('hourFilter', 'value'),
              [Input('resetFilter', 'n_clicks'),
               Input('dropdown', 'value'),]
              )
def resetFilterFunction(n_clicks, uuid):
    nSamplesPerHour = dataF[uuid]['nSamplesPerHour']
    hourValue = [0, 24*nSamplesPerHour-1]
    return hourValue

@app.callback(
              Output('ampFilter', 'value'),
              [Input('resetFilter', 'n_clicks'),
               Input('dropdown', 'value'),]
              )
def resetFilterFunction(n_clicks, uuid):
    minenergy = dataF[uuid]['minenergy']
    maxenergy = dataF[uuid]['maxenergy']
    ampValue = [minenergy, maxenergy]
    return ampValue

@app.callback(
              Output('maxAmpFilter', 'value'),
              [Input('resetFilter', 'n_clicks'),
               Input('dropdown', 'value'),]
              )
def resetFilterFunction(n_clicks, uuid):
    minenergy = dataF[uuid]['minenergy']
    maxenergy = dataF[uuid]['maxenergy']
    ampValue = [minenergy, maxenergy]
    return ampValue

@app.callback(
              Output('dateFilter', 'value'),
              [Input('resetFilter', 'n_clicks'),
               Input('dropdown', 'value'),]
              )
def resetFilterFunction(n_clicks, uuid):
    date_dt = dataF[uuid]['date_dt']
    dateRange = [0, len(date_dt)-1]
    return dateRange

@app.callback(
              Output('toggle', 'value'),
              [Input('resetFilter', 'n_clicks'),]
              )
def resetFilterFunction(n_clicks):
    toggleValue = 'New File'
    return toggleValue

@app.callback(
    Output('dailyPlot', 'figure'),
    [Input('clickable-graph', 'hoverData'),
     Input('ampFilter', 'value'),
     Input('maxAmpFilter', 'value'),
     Input('hourFilter', 'value'),
     Input('dailyConsFilter', 'value'),
     Input('dateFilter', 'value'),
     Input('dropdown', 'value'),
     ])
def updateDailyPlot(hoverData, ampValue, maxAmpValue, hourValue, dailyConsValue, dateRange, uuid):
    dates = dataF[uuid]['dates']
    dailyConsumption = dataF[uuid]['dailyConsumption']
    dailyMax = dataF[uuid]['dailyMaxConsumption']
    df = dataF[uuid]['df']
    ymin = dataF[uuid]['ymin']
    ymax = dataF[uuid]['ymax']
    original_datelist = dataF[uuid]['original_datelist']
    date_dt = dataF[uuid]['date_dt']
    
    row = pd.DataFrame(hoverData['points'])
    if 'y' in row.columns:
        STARTING_DATE = row['y'].iloc[0]
    else:
        STARTING_DATE = dates[0]

    validDates = dailyConsumption[(dailyConsumption['Value'] >= dailyConsValue[0]) & (dailyConsumption['Value'] <= dailyConsValue[1]) & (dailyMax['Value'] >= maxAmpValue[0]) & (dailyMax['Value'] <= maxAmpValue[1])].Date
    DAILY_IMG3 = df.loc[((original_datelist >= date_dt[dateRange[0]]) & (original_datelist <= date_dt[dateRange[1]]) & (df['Date'] == STARTING_DATE) & (df['Hour'] >= hourValue[0]/2.0) & (df['Hour'] <= hourValue[1]/2.0) & (df['Value'] >= ampValue[0]) & (df['Value'] <= ampValue[1]) & (original_datelist.isin(validDates)))][['Fractional_hour','Hour','Value']]
    ymax_new = int(min(ymax, ampValue[1]*nSamplesPerHour))
    ymin_new = int(max(ymin, ampValue[0]*nSamplesPerHour))
    data = [ dict(
                  x = DAILY_IMG3['Fractional_hour'],
                  y = DAILY_IMG3['Value']*nSamplesPerHour,
                  mode = 'lines+markers',
                  ) ]
    layout = dict(
                  title=STARTING_DATE,
                  xaxis = dict(title='Hours', nticks=8, range = [0,24],),
                  yaxis = dict(title='Power (W)', ticks='', range = [ymin_new, ymax_new] )
                  )
    return dict( data=data, layout=layout )

@app.callback(
              Output('histogram-graph', 'figure'),
              [Input('dailyConsFilter', 'value'),
               Input('dateFilter', 'value'),
               Input('dropdown', 'value'),
               Input('options', 'value'),
               Input('clickable-graph', 'clickData'),
               ])
def update_histogram(dailyConsValue, dateRange, uuid, option_value, click_data):
    dailyConsumption = dataF[uuid]['dailyConsumption']
    date_dt = dataF[uuid]['date_dt']
    vacation_dates = pd.to_datetime(dataF[uuid]['vacation_dates'], format='%Y-%m-%d')
    dailyConsumption['Date'] = pd.to_datetime(dailyConsumption['Date'], format='%Y-%m-%d')
    if (option_value == 'vacation'):
        validDailyConsumption = dailyConsumption[(dailyConsumption['Value'] >= dailyConsValue[0]) & (dailyConsumption['Value'] <= dailyConsValue[1]) & (dailyConsumption['Date'] >= date_dt[dateRange[0]]) & (dailyConsumption['Date'] <= date_dt[dateRange[1]]) & (date_dt.isin(vacation_dates))]
    elif (option_value == 'non-vacation'):
        validDailyConsumption = dailyConsumption[(dailyConsumption['Value'] >= dailyConsValue[0]) & (dailyConsumption['Value'] <= dailyConsValue[1]) & (dailyConsumption['Date'] >= date_dt[dateRange[0]]) & (dailyConsumption['Date'] <= date_dt[dateRange[1]]) & ~(date_dt.isin(vacation_dates))]
    else:
        validDailyConsumption = dailyConsumption[((dailyConsumption['Value'] >= dailyConsValue[0]) & (dailyConsumption['Value'] <= dailyConsValue[1]) & (dailyConsumption['Date'] >= date_dt[dateRange[0]]) & (dailyConsumption['Date'] <= date_dt[dateRange[1]]))]
    
    data = [go.Histogram(
                  x = validDailyConsumption['Value'],
                  )]
    layout = [dict(
                  title='Heatmap of daily consumption (KwH)',
                  height = 400,
                  xaxis = dict(title='KwH', ticks=''),
                   yaxis = dict(title='# occurrences', ticks='' )
                  )]
    return dict( data=data, layout=layout )

@app.callback(
              Output('clickable-graph', 'figure'),
              [Input('ampFilter', 'value'),
               Input('maxAmpFilter', 'value'),
               Input('hourFilter', 'value'),
               Input('dailyConsFilter', 'value'),
               Input('dateFilter', 'value'),
               Input('dropdown', 'value'),
               Input('options', 'value'),
               Input('clickable-graph', 'clickData'),
               Input('vac_togggler', 'value'),
               Input('reset-vac-data', 'value')])
def update_graph(ampValue,maxAmpValue,hourValue,dailyConsValue,dateRange,uuid, option_value, click_data, vacation_toggler, vacation_reset):
    if vacation_reset:
        dataF[uuid]['vacation_dates'] = dataF[uuid]['vacation_dates_backup']
    if ((click_data is not None) & (vacation_toggler == 'enable_click')):
        row = pd.DataFrame(click_data['points'])
        CLICKED_DATE = row['y'].iloc[0]
        CLICKED_x   = row['x'].iloc[0]
        if dataF[uuid]['click_data'] is not None:
            if ((CLICKED_x != dataF[uuid]['click_data']['x'].iloc[0]) | (CLICKED_DATE != dataF[uuid]['click_data']['y'].iloc[0])):
                if (option_value == 'vacation'):
                    index_to_delete = [(i != str(CLICKED_DATE)) for i in dataF[uuid]['vacation_dates']]
                    dataF[uuid]['vacation_dates'] = [e for i, e in enumerate(dataF[uuid]['vacation_dates']) if e != str(CLICKED_DATE)]
                else:
                    dataF[uuid]['vacation_dates'] = np.hstack((dataF[uuid]['vacation_dates'], CLICKED_DATE))
        dataF[uuid]['click_data'] = row
    B = np.copy(dataF[uuid]['A'])
    hours = dataF[uuid]['hours']
    dates = dataF[uuid]['dates']
    ymin = dataF[uuid]['ymin']
    ymax = dataF[uuid]['ymax']
    vacation_dates = dataF[uuid]['vacation_dates']

    dailySum = np.nansum(B, axis=1)/B.shape[1]*24/1000
    dailyMax = np.nanmax(B, axis=1)
    if(np.nanmax(B) > ampValue[1]):
        B[B > ampValue[1]] = np.nan
    if(np.nanmin(B) < ampValue[0]):
        B[B < ampValue[0]] = np.nan
    
    B[:,0:hourValue[0]] = np.nan
    B[:,hourValue[1]:] = np.nan
    
    validDates = ((dailySum >= dailyConsValue[0]) & (dailySum <= dailyConsValue[1]) & (dailyMax >= maxAmpValue[0]) & (dailyMax <= maxAmpValue[1]))

    validDates[0:dateRange[0]] = False
    validDates[dateRange[1]:] = False

    if (option_value == 'vacation'):
        alldates_df = pd.DataFrame(dates, columns = ['date'])
        vac_df = pd.DataFrame(vacation_dates, columns = ['date'])
        index_test = alldates_df['date'].isin(vac_df['date'])
        validDates[np.where(index_test == False)] = False
        dates = dates[validDates]
    if (option_value == 'non-vacation'):
        alldates_df = pd.DataFrame(dates, columns = ['date'])
        vac_df = pd.DataFrame(vacation_dates, columns = ['date'])
        index_test = alldates_df['date'].isin(vac_df['date'])
        validDates[np.where(index_test == True)] = False
        dates = dates[validDates]

    B[~validDates,:] = np.nan
    data = [go.Heatmap(
                  z=B,
                  x=hours,
                  y=dates,
                  colorscale='Jet'
              )]
    layout = [dict(
                  title='Heatmap',
                  height = 800,
                  xaxis = dict(ticks = '', range = [0,23],),
                  yaxis = dict(ticks='', range = [ymin, ymax] )
                  )]

    if ((option_value == 'vacation') | (option_value == 'non-vacation')):
        data = [go.Heatmap(
                           z=B[validDates,:],
                           x=hours,
                           y=dates,
                           colorscale='Jet'
                           )]
        layout = [dict(
                       title='Heatmap',
                       height = 800,
                       xaxis = dict(ticks = '', range = [0,23],),
                       yaxis = dict(ticks='', range = [] )
                       )]
    return dict( data=data, layout=layout )

def filter_data(value):
    if value == 'all':
        return df
    else:
        return [df['Date'] == value]


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "//fonts.googleapis.com/css?family=Raleway:400,300,600",
                "//fonts.googleapis.com/css?family=Dosis:Medium",
                "https://cdn.rawgit.com/plotly/dash-app-stylesheets/0e463810ed36927caf20372b6411690692f94819/dash-drug-discovery-demo-stylesheet.css"]


for css in external_css:
    app.css.append_css({"external_url": css})


if __name__ == '__main__':
    app.run_server()
