from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django_pandas.managers import DataFrameManager
from fcm_django.models import FCMDevice
from django.db.models import F


from .models import PlantEnviron, WeatherForecast, Prediction

import re
import pandas as pd
import numpy as np
from datetime import timedelta
from plotly.offline import plot #plotly
from plotly.graph_objs import *
import plotly.graph_objs as go
from plotly.subplots import make_subplots



from django.core import mail
from django.core.mail import EmailMessage
from django.template.loader import render_to_string
from django.utils.html import strip_tags
from django.http import HttpResponse



@login_required
def index(request):
    context = {
        'plant': 0
    }
    return render(request, 'monitor/index.html', context)


@login_required
def plant(request, plant):
    env = PlantEnviron.objects.order_by(F('recTime').desc()).filter(plant=plant).first()
    pred24 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='24')[:24]
    pred48 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='48')[:24]
    cond24 = pred24.to_dataframe()
    cond24 = cond24.filter(regex='cond').apply(lambda x: any(x))
    cond48 = pred48.to_dataframe()
    cond48 = cond48.filter(regex='cond').apply(lambda x: any(x))

    context = {
        'plant': plant,
        'env': env,
        'pred24': pred24,
        'pred48': pred48,
        'cond24': cond24,
        'cond48': cond48,
    }


    return render(request, 'monitor/plant.html', context)


@login_required
def loc(request, plant, loc):

    # 현재 공장 환경 (차트)
    # 공장 환경 : 현재 값-24시간 + 현재 기준으로 24, 48예측 값 (plotly)
    env = PlantEnviron.objects.order_by(F('recTime').desc()).filter(plant=plant)[:24]
    pred24 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='24')[:24]
    pred48 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='48')[:24]

    env_df = env.to_dataframe().sort_values(by=['recTime'])
    pred24_df = pred24.to_dataframe().sort_values(by=['recTime'])
    pred24_df['recTime'] = pred24_df['recTime'] + pd.offsets.Hour(24)
    pred48_df = pred48.to_dataframe().sort_values(by=['recTime'])
    pred48_df['recTime'] = pred48_df['recTime'] + pd.offsets.Hour(48)

    #(마지막 line그래프에서 써야함)

    # #loc 선택
    filters = f'((in|coil|cond)_loc{loc}|out|recTime)'
    env_df = env_df.filter(regex=filters)
    pred24_df = pred24_df.filter(regex=filters)
    pred48_df = pred48_df.filter(regex=filters)
    env_df.columns = [re.sub('\d', '', col) for col in env_df.columns]
    pred24_df.columns = [re.sub('\d', '', col) for col in pred24_df.columns]
    pred48_df.columns = [re.sub('\d', '', col) for col in pred48_df.columns]

    #이슬점
    def dewpoint(temp, humid):
        return ((243.12 *((17.62 * temp /(243.12 + temp)) + np.log(humid / 100.0))) 
            / (17.62-((17.62 * temp / (243.12 + temp)) + np.log(humid/ 100.0))))

    tem_col = env_df.filter(regex='tem_in_').columns
    hum_col = env_df.filter(regex='hum_in_').columns

    dew_col = f'{tem_col[0][:3]}_dewpoint_{tem_col[0][-6:]}'
    print(dew_col)
    env_df[dew_col] = dewpoint(env_df[tem_col[0]], env_df[hum_col[0]])
    pred24_df[dew_col] = dewpoint(pred24_df[tem_col[0]], pred24_df[hum_col[0]])
    pred48_df[dew_col] = dewpoint(pred48_df[tem_col[0]], pred48_df[hum_col[0]])

    # #시각화
    #이중축있는 subplot
    fig = make_subplots(specs=[[{"secondary_y": True}]])    

    fig.add_trace(go.Scatter(x=env_df['recTime'], y=env_df['tem_in_loc'],  mode='lines+markers', name='온도',
                             opacity=0.8, marker_color='red'), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred24_df['recTime'], y=pred24_df['tem_in_loc'], mode='lines+markers', name='24시간후 온도',
                             opacity=0.8, line = dict(color='red', width=3, dash='dash')), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred48_df['recTime'], y=pred48_df['tem_in_loc'], mode='lines+markers', name='48시간후 온도',
                             opacity=0.8,line = dict(color='red', width=3, dash='dot')), secondary_y = False)

    fig.add_trace(go.Scatter(x=env_df['recTime'], y=env_df['tem_coil_loc'], mode='lines+markers',  name='코일온도',
                             opacity=0.8, marker_color='rgb(245,73,19)'), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred24_df['recTime'], y=pred24_df['tem_coil_loc'], mode='lines+markers', name='24시간후 코일온도',
                             opacity=0.8, line = dict(color='rgb(245,73,19)', width=3, dash='dash')), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred48_df['recTime'], y=pred48_df['tem_coil_loc'], mode='lines+markers', name='48시간후 코일온도',
                             opacity=0.8, line = dict(color='rgb(245,73,19)', width=3, dash='dot')), secondary_y = False)

    fig.add_trace(go.Scatter(x=env_df['recTime'], y=env_df['tem_dewpoint_in_loc'], mode='lines+markers',  name='이슬점',
                             opacity=0.8, marker_color='rgb(48,85,152)'), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred24_df['recTime'], y=pred24_df['tem_dewpoint_in_loc'], mode='lines+markers', name='24시간후 이슬점',
                             opacity=0.8,  line = dict(color='rgb(48,85,152)', width=3, dash='dash')), secondary_y = False)
    fig.add_trace(go.Scatter(x=pred48_df['recTime'], y=pred48_df['tem_dewpoint_in_loc'], mode='lines+markers', name='48시간후 이슬점',
                             opacity=0.8, line = dict(color='rgb(48,85,152)', width=3, dash='dot')), secondary_y = False)

    fig.add_trace(go.Bar(x=env_df['recTime'], y=env_df['hum_in_loc'],  name='습도',
                             opacity=0.8, marker_color='skyblue'), secondary_y = True)
    fig.add_trace(go.Bar(x=pred24_df['recTime'], y=pred24_df['hum_in_loc'],  name='24시간후 습도',
                             opacity=0.8, marker_color='skyblue'), secondary_y = True)
    fig.add_trace(go.Bar(x=pred48_df['recTime'], y=pred48_df['hum_in_loc'],  name='48시간후 습도',
                             opacity=0.8, marker_color='skyblue'), secondary_y = True)
    

    fig.update_layout(title='<b>Today Factory Environment</b>',barmode='overlay', yaxis2=dict(showgrid=False, zeroline=False))
    fig.update_traces(yaxis='y2', selector={'type':'bar'}, marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.0, opacity=0.5)
    fig.update_xaxes(title_text='<b>날짜</b>')
    fig.update_yaxes(title_text='<b>[ 온도(ºC) ]</b>')
    fig.update_yaxes(title_text='<b>[ 상대습도(%) ]</b>', secondary_y=True)

    plot_div = plot(fig, output_type='div')

    # TODO : 현재 ~ 24시간까지 결로 여부, 24~48시간 이내 결로 여부

    context = {
        'plant': plant,
        'loc': loc,
        'plant_envs': env_df.iloc[-1], #환경 CHART
        'plot_div': plot_div,
        'pred24': pred24_df.iloc[-1],
        'pred48': pred48_df.iloc[-1],
        # 'x': x_data,
        # 'y': y_data,
    }
    return render(request, 'monitor/loc.html', context)



def visualization(request):

    #현재값 + 예측값
    #plant_envs = Prediction.objects.all().order_by('recTime').last()
    plant_all = Prediction.objects.all()
    cnt = len(plant_all)-1
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
            'plant_envs' : plant_envs,
            'plot_div': plot_div,
            'x' : x_data,
            'y' : y_data,
        }

    return render(request, 'monitor/visualization.html', context)


def notification(request):
    device = FCMDevice.objects.all().first()
    device.send_message("title", "this is test message")
    context = {
        'plant': 0
    }

    return render(request, 'monitor/index.html', context)

def sendMail(request, plant):
    env = PlantEnviron.objects.order_by(F('recTime').desc()).filter(plant=plant).first()
    pred24 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='24')[:24]
    pred48 = Prediction.objects.order_by(F('recTime').desc()).filter(plant=plant).filter(forecast='48')[:24]
    cond24 = pred24.to_dataframe()
    cond24 = cond24.filter(regex='cond').apply(lambda x: any(x))
    cond48 = pred48.to_dataframe()
    cond48 = cond48.filter(regex='cond').apply(lambda x: any(x))

    context = {
        'plant' : plant,
        'env': env,
        'pred24': pred24,
        'pred48': pred48,
        'cond24': cond24,
        'cond48': cond48,
    }

    mail_title = '결로 발생 경보'
    html_text = render_to_string('monitor/mail.html',context)
    admins = ['reqip95@gmail.com']

    email = EmailMessage(
        mail_title,
        html_text,
        to=admins,
    )
    email.content_subtype = 'html'
    email.send()
    return HttpResponse('Mail successfully sent')
