from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from plotly.offline import plot #plotly
from plotly.graph_objs import Scatter
# Create your views here.

@login_required
def index(request):
    context = {
        
    }
    return render(request, 'monitor/index.html', context)

def line(request):
    x_data = [0,1,2,3,4]
    y_data = [x**2 for x in x_data]
    plot_div = plot([Scatter(x=x_data, y=y_data, mode='lines', name='test',
    opacity=0.8, marker_color='green')],output_type='div')
    return render(request, 'monitor/index.html', context={'plot_div': plot_div})
