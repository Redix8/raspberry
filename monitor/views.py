from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from plotly.offline import plot #plotly
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from django_pandas.managers import DataFrameManager
import pandas as pd

from .models import PlantEnviron, WeatherForecast, Prediction
from IPython import embed
# Create your views here.

@login_required
def index(request):
    context = {
        'plant': 0
    }

    return render(request, 'monitor/index.html', context)

@login_required
def plant(request, plant):
    context = {
        'plant': plant,
    }
    return render(request, 'monitor/plant.html', context)


@login_required
def loc(request, plant, loc):

    # 현재값 + 예측값
    # plant_envs = Prediction.objects.all().order_by('recTime').last()
    plant_all = Prediction.objects.all().filter(plant=plant)
    cnt = len(plant_all) - 1
    plant_envs = plant_all[cnt]

    # 공장 환경 : 현재 값-24시간 + 현재 기준으로 24, 48예측 값
    env_df = PlantEnviron.objects.all().filter(plant=plant).to_dataframe() 
    y_data = env_df.iloc[-24:, 1:]
    x_data = env_df.iloc[-24:, 1] #시간

    print(x_data)
    print(y_data)

    filters= f'(tem|hum)' + '_(in|coil)' + '_loc' + str(loc)
    y_data = y_data.filter(regex=filters, axis=1) # loc1? temp_in_loc1, hum_in_loc1, coil_in_loc1

    #24, 48시간 예측값
    objects = DataFrameManager()
    pred = Prediction.objects.all().to_dataframe()
    # print(pred)
    pred = pred.iloc[-48:,:] # 현재 관측값 기준 -24Hours
    # embed()
    if (pred['forecast'] == '24').bool:
        pred_24 = pred[pred['forecast'] == '24'] #24시간후 예측값 df
        pred_24 = pred_24.filter(regex=filters, axis=1)
    elif (pred['forecast'] == '48').bool:
        pred_48 = pred[pred['forecast'] == '48'] #48시간후 예측값 df
        pred_48 = pred_48.filter(regex=filters, axis=1)

    index_24 = pd.date_range(start=pred_24.index[-1], periods=24, freq='1H')
    # index_48 = pd.date_range(start=pred_48.index[-1], periods=48, freq='1H')

    plant_24 = pd.DataFrame({'index':index_24}).join(pred_24)
    # plant_48 = pd.DataFrame({'index':index_48}).join(pred_48)

    #시각화
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data.iloc[:,0], mode='lines+markers', name='온도',
                             opacity=0.8, marker_color='red'))

    fig.add_trace(go.Scatter(x=plant_24.index, y=plant_24.iloc[:,0], mode='lines+markers', name='24시간후 온도',
                             opacity=0.8, marker_color='green'))

    # fig.add_trace(go.Scatter(x=plant_48.index, y=plant_48.iloc[:,0], mode='lines+markers', name='48시간후 온도',
    #                          opacity=0.8, marker_color='orange'))

    fig.update_layout(title='<b>Today Factory Environment</b>', xaxis_title='Date', yaxis_title='Scale')

    plot_div = plot(fig, output_type='div')

    context = {
        'plant': plant,
        'loc': loc,
        'plant_envs': plant_envs,
        'plot_div': plot_div,
        'x': x_data,
        'y': y_data,
    }
    return render(request, 'monitor/loc.html', context)


def visualization(request):

    #현재값 + 예측값
    #plant_envs = Prediction.objects.all().order_by('recTime').last()
    plant_all = Prediction.objects.all()
    cnt = len(plant_all)-1
    plant_envs = plant_all[cnt]

    
   
    # 공장 환경 : 현재 값-24시간 + 현재 기준으로 24, 48예측 값
    objects = DataFrameManager()
    envs = PlantEnviron.objects.all().filter(plant=plant)  
    env_df = envs.to_dataframe()
    y_data = env_df.iloc[:24, 1:]
    x_data = env_df.iloc[:24, :1] #시간
    print(x_data)
    print(y_data)
    filters= f'(tem|hum)' + '_(in|coil)' + '_loc' + str(loc)

    y_data = y_data.filter(regex=filters, axis=1)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data.iloc[:,0], mode='lines+markers', name='온도',
                             opacity=0.8, marker_color='red'))

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data.iloc[:,1], mode='lines+markers', name='습도',
                             opacity=0.8, marker_color='green'))

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data.iloc[:,2], mode='lines+markers', name='코일 온도',
                             opacity=0.8, marker_color='orange'))

    fig.update_layout(title='<b>Today Factory Environment</b>', xaxis_title='Date', yaxis_title='Scale')

    plot_div = plot(fig, output_type='div')

    context = {
            'plant_envs' : plant_envs,
            'plot_div': plot_div,
            'x' : x_data,
            'y' : y_data,
        }
    return render(request, 'monitor/visualization.html', context)
