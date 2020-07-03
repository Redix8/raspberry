from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from plotly.offline import plot #plotly
from plotly.graph_objs import *
import plotly.graph_objs as go

from .models import PlantEnviron, WeatherForecast, Prediction
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
    context = {
        'plant': plant,
        'loc': loc,
    }
    return render(request, 'monitor/loc.html', context)


def visualization(request):

    #현재값 + 예측값
    #plant_envs = Prediction.objects.all().order_by('recTime').last()
    plant_all = Prediction.objects.all()
    cnt = len(plant_all)-1
    plant_envs = plant_all[cnt]

    
    # 공장 환경 : 현재 값-24시간 + 현재 기준으로 24, 48예측 값
    envs = PlantEnviron.objects.all() 
    # sample -> object 1개 -> sample.plant -> 1 / 2
    #env_df = read_frame(envs, fieldnames=['recTime', 'tem_in_loc1','hum_in_loc1','tem_coil_loc1'])
    env_df = envs.to_dataframe(['recTime','tem_in_loc1','hum_in_loc1','tem_coil_loc1'])
    # env_df['fcTime'] = env_df['fcTime'].to_datetime()
    x_data = env_df.iloc[:24,:1]
    y_data = env_df.iloc[:24,1:]
    # print(x_data)
    # print(y_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data.index, y=y_data['tem_in_loc1'], mode='lines+markers', name='온도',
                 opacity=0.8, marker_color='red'))

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data['hum_in_loc1'], mode='lines+markers', name='습도',
                 opacity=0.8, marker_color='green'))

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data['tem_coil_loc1'],mode='lines+markers', name='코일 온도',
                 opacity=0.8))
    fig.update_layout(title='<b>Today Factory Environment</b>', xaxis_title='Date', yaxis_title='Scale')          

    plot_div = plot(fig, output_type='div')

    context = {
            'plant_envs' : plant_envs,
            'plot_div': plot_div,
            'x' : x_data,
            'y' : y_data,
        }
    return render(request, 'monitor/visualization.html', context)
