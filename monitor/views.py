from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from plotly.offline import plot #plotly
from plotly.graph_objs import *
import plotly.graph_objs as go

from .models import PlantEnviron, WeatherForecast
# Create your views here.


@login_required
def index(request):
    context = {
        
    }
    return render(request, 'monitor/index.html', context)


@login_required
def plant1(request):
    context = {

    }
    return render(request, 'monitor/plant1.html', context)


def line(request):
    # articles = Article.objects.all()
    envs = PlantEnviron.objects.all() 
    # sample -> object 1개 -> sample.plant -> 1 / 2
    #env_df = read_frame(envs, fieldnames=['recTime', 'tem_in_loc1','hum_in_loc1','tem_coil_loc1'])
    env_df = envs.to_dataframe(['recTime','temp_25','humid_25','rain_25'])
    # env_df['fcTime'] = env_df['fcTime'].to_datetime()
    x_data = env_df.iloc[:24,:1]
    y_data = env_df.iloc[:24,1:]
    print(x_data)
    print(y_data)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_data.index, y=y_data['tem_in_loc1'], mode='lines+markers', name='온도',
                 opacity=0.8, marker_color='red'))

    fig.add_trace(go.Scatter(x=x_data.index, y=y_data['hum_in_loc1'], mode='lines+markers', name='습도',
                 opacity=0.8, marker_color='green'))

    fig.add_trace(go.Bar(x=x_data.index, y=y_data['tem_coil_loc1'], name='강수확률',
                 opacity=0.8))
    fig.update_layout(title='<b>공장 환경 관측</b>', xaxis_title='Date', yaxis_title='Scale')          

    plot_div = plot(fig, output_type='div')
    context = {
            'plot_div': plot_div,
            'x' : x_data,
            'y' : y_data,
        }
    return render(request, 'monitor/line.html', context)
