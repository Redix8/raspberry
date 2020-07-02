from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from plotly.offline import plot #plotly
from plotly.graph_objs import *
import plotly.plotly as py

from .models import PlantEnviron, WeatherForecast
# Create your views here.

@login_required
def index(request):
    context = {
        
    }
    return render(request, 'monitor/index.html', context)

def line(request):
    # articles = Article.objects.all()
    envs = WeatherForecast.objects.all() 
    # sample -> object 1개 -> sample.plant -> 1 / 2
    #env_df = read_frame(envs, fieldnames=['recTime', 'tem_in_loc1','hum_in_loc1','tem_coil_loc1'])
    env_df = envs.to_dataframe(['fcTime','temp_25','humid_25','rain_25'])
    # for sample in samples:
    #     x = sample.lc[2:5]
    #     y = sample[6:]
    x_data = env_df.iloc[:24,:1]
    y_data = env_df.iloc[:24,1:]

    layout = Layout(plot_bgcolor='rgba(0,0,0,0)')
    data = Data([Scatter(x=x_data.index, y=y_data['temp_25'], mode='lines', name='test',
    opacity=0.8, marker_color='green')], output_type='div')
    # context = {
    #     'plot_div': plot_div,
    # }
    fig = Figure(data=data, layout=layout)
    plot(fig, output_type='div')
    return render(request, 'monitor/line.html', context)
