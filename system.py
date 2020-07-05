import os
import pickle
import numpy as np
import pandas as pd
import django
import time
from django_pandas.io import read_frame
import datetime
from copy import deepcopy
import warnings
from tqdm import tqdm

warnings.filterwarnings(action="ignore")

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raspberry.settings")
django.setup()
from fcm_django.models import FCMDevice
from monitor.models import PlantEnviron, WeatherForecast, Prediction

def dewpoint(temp, humid):
    return ((243.12 *((17.62 * temp /(243.12 + temp)) + np.log(humid / 100.0))) 
            / (17.62-((17.62 * temp / (243.12 + temp)) + np.log(humid/ 100.0))))


with open('models/forecast_model.bin', 'rb') as f:
    forecasters = pickle.load(f)

with open('models/scalers.bin', 'rb') as f:
    scalers = pickle.load(f)

with open('models/classifiers.bin','rb') as f:
    classifiers = pickle.load(f)

plant1_new = pd.read_csv('.data/plant1_test_split.csv', parse_dates=[0])
plant2_new = pd.read_csv('.data/plant2_test_split.csv', parse_dates=[0])
forecast = read_frame(WeatherForecast.objects.all())
Prediction.objects.all().delete()

for idx in tqdm(range(len(plant1_new))):
    start_time = time.time()
    # 새로 들어온 데이터
    new1 = plant1_new.iloc[idx, :]
    new2 = plant2_new.iloc[idx, :]

    # 새 데이터 db 저장
    col = ['recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2', 
    'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3', 
    'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3', ]
    new1.index = col
    new1.recTime = pd.to_datetime(new1.recTime)

    df1 = pd.DataFrame(columns=col)
    df1 = df1.append(new1)
    df1.index = df1.recTime
    df1['plant'] = '1'

    new2.index = col
    new2.recTime = pd.to_datetime(new2.recTime)

    df2 = pd.DataFrame(columns=col)
    df2 = df2.append(new2)
    df2.index = df2.recTime
    df2['plant'] = '2'

    PlantEnviron.objects.bulk_create(
    PlantEnviron(**vals) for vals in df1.to_dict('records')
    )

    PlantEnviron.objects.bulk_create(
    PlantEnviron(**vals) for vals in df2.to_dict('records')
    )

    # 날씨 데이터( 전처리 했다고 가정하고 DB에서 불러온다)

    fc_col = ['fcTime', 'temp_25', 'temp_46', 'humid_25', 'humid_46', 'rain_25', 'rain_46', 'wind_25', 'wind_46']
    fc = pd.DataFrame(columns=fc_col)

    rectime = new1.recTime
    forecasts = WeatherForecast.objects.filter(fcTime__range=(rectime - datetime.timedelta(hours=24),rectime))
    for obj in forecasts:
        data = pd.Series({'fcTime':obj.fcTime, 'temp_25':obj.temp_25, 'temp_46':obj.temp_46,
        'humid_25':obj.humid_25, 'humid_46':obj.humid_46, 'rain_25':obj.rain_25, 'rain_46':obj.rain_46,
        'wind_25':obj.wind_25, 'wind_46':obj.wind_46})
        fc = fc.append(data, ignore_index=True)
    
    fc.index = fc.fcTime
    fc.drop('fcTime',axis=1,inplace=True)

    # # 1차 예측을 위해서 데이터 처리(feature 생성)
    data  = pd.concat([df1,fc], axis=1)
    data2 = pd.concat([df2,fc], axis=1)

    test_X = data.drop(['recTime','cond_loc1', 'cond_loc2', 'cond_loc3'], axis=1)
    ma6 =  test_X.rolling(6).mean().filter(regex='(25|46)').add_prefix('MA6_')
    ma24 = test_X.rolling(24).mean().filter(regex='(25|46)').add_prefix('MA24_')
    test_X = pd.concat([test_X, ma6, ma24], axis=1).dropna()

    test2_X = data2.drop(['recTime','cond_loc1', 'cond_loc2', 'cond_loc3'], axis=1)
    ma6 =  test2_X.rolling(6).mean().filter(regex='(25|46)').add_prefix('MA6_')
    ma24 = test2_X.rolling(24).mean().filter(regex='(25|46)').add_prefix('MA24_')
    test2_X = pd.concat([test2_X, ma6, ma24], axis=1).dropna()

    # 1차 예측
    # plant 1
    plant1_pred_step1 = pd.DataFrame()
    for col in forecasters['1']:
        preds = []
        time_col = test_X.filter(regex= f'{col[1:3]}$').columns.to_list()        
        in_col = test_X.filter(regex=f'(in|coil)_loc{col[-1]}').columns.to_list()
        out_col = test_X.filter(regex=f'out_loc1').columns.to_list()
        if 'out_loc1' in col:
            tcol = time_col + out_col
        else:
            tcol = time_col + in_col

        x = test_X[tcol]
        scaler = scalers['1'][col]
        x.loc[:,:] = scaler.transform(x)

        for model in forecasters['1'][col]:        
            preds.append(model.predict(x))    
        pred = np.mean(preds, axis=0)
        plant1_pred_step1[col] = pred  
    plant1_pred_step1.index = test_X.index

    # plant 2
    plant2_pred_step1 = pd.DataFrame()
    for col in forecasters['2']:
        preds = []
        time_col = test_X.filter(regex= f'{col[1:3]}$').columns.to_list()        
        in_col = test_X.filter(regex=f'(in|coil)_loc{col[-1]}').columns.to_list()
        out_col = test_X.filter(regex=f'out_loc1').columns.to_list()
        if 'out_loc1' in col:
            tcol = time_col + out_col
        else:
            tcol = time_col + in_col

        x = test_X[tcol]
        scaler = scalers['2'][col]
        x.loc[:,:] = scaler.transform(x)

        for model in forecasters['2'][col]:        
            preds.append(model.predict(x))    
        pred = np.mean(preds, axis=0)
        plant2_pred_step1[col] = pred  
    plant2_pred_step1.index = test_X.index

    # 1차 예측으로 2차 예측 시작
    # 2차 예측을 위해서 데이터 처리(feature 생성)
    plant1_pred = deepcopy(plant1_pred_step1)
    plant2_pred = deepcopy(plant2_pred_step1)

    # plant1 전처리
    tem_col = plant1_pred.filter(regex='tem_in_').columns
    hum_col = plant1_pred.filter(regex='hum_in_').columns
    coil_col = plant1_pred.filter(regex='coil_').columns

    for i in range(len(tem_col)):
        dew_col = f'{tem_col[i][:3]}_dewpoint_{tem_col[i][-7:]}'
        plant1_pred[dew_col] = dewpoint(plant1_pred[tem_col[i]], plant1_pred[hum_col[i]])

        plant1_pred[f'{tem_col[i][:3]}_dewdiff_{tem_col[i][-7:]}'] = plant1_pred[coil_col[i]] - plant1_pred[dew_col]

    plant1_pred['month'] = plant1_pred.index.month
    plant1_pred['day'] = plant1_pred.index.day
    plant1_pred['hour'] = plant1_pred.index.hour

    # plant2 전처리
    tem_col = plant2_pred.filter(regex='tem_in_').columns
    hum_col = plant2_pred.filter(regex='hum_in_').columns
    coil_col = plant2_pred.filter(regex='coil_').columns

    for i in range(len(tem_col)):
        dew_col = f'{tem_col[i][:3]}_dewpoint_{tem_col[i][-7:]}'
        plant2_pred[dew_col] = dewpoint(plant2_pred[tem_col[i]], plant2_pred[hum_col[i]])

        plant2_pred[f'{tem_col[i][:3]}_dewdiff_{tem_col[i][-7:]}'] = plant2_pred[coil_col[i]] - plant2_pred[dew_col]

    plant2_pred['month'] = plant2_pred.index.month
    plant2_pred['day'] = plant2_pred.index.day
    plant2_pred['hour'] = plant2_pred.index.hour

    # 2차 예측 시행
    # plant1 결로 예측
    test_pred = {}
    for time_label in ['y25', 'y46']:
        X_time = plant1_pred.filter(regex=f'{time_label}')
        for loc_label in ['loc1', 'loc2', 'loc3']:
            in_col = X_time.filter(regex=f'(in|coil)_{loc_label}').columns.to_list()
            out_col = X_time.filter(regex=f'out_loc1').columns.to_list()
            date_col = ['month','day', 'hour']
            tcol = in_col + out_col + date_col

            p = np.zeros(X_time.shape[0])
            for m in classifiers['1'][f'{time_label}_{loc_label}']:
                p += (m.predict_proba( plant1_pred[tcol] )/5)[:, 1].reshape(-1,)
                p_cond = np.where(p>0.3, 1, 0)
            test_pred[f'{loc_label}_{time_label}'] = p_cond

    # plant2 결로 예측
    test2_pred = {}
    for time_label in ['y25', 'y46']:
        X_time = plant2_pred.filter(regex=f'{time_label}')
        for loc_label in ['loc1', 'loc2', 'loc3']:
            in_col = X_time.filter(regex=f'(in|coil)_{loc_label}').columns.to_list()
            out_col = X_time.filter(regex=f'out_loc1').columns.to_list()
            date_col = ['month','day', 'hour']
            tcol = in_col + out_col + date_col

            p = np.zeros(X_time.shape[0])
            for m in classifiers['2'][f'{time_label}_{loc_label}']:
                p += (m.predict_proba( plant2_pred[tcol] )/5)[:, 1].reshape(-1,)[0]
                p_cond = np.where(p>0.3, 1, 0)
            test2_pred[f'{loc_label}_{time_label}'] = p_cond

    # 1차, 2차 예측 DB 저장
    f_cols = ['recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1',
        'tem_in_loc2', 'hum_in_loc2', 'tem_coil_loc2',
        'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3',
        'tem_out_loc1', 'hum_out_loc1'
    ]
    plant1_save_24 = plant1_pred_step1.filter(regex='y25')
    plant1_save_24 = plant1_save_24.reset_index()
    plant1_save_24.columns = f_cols
    plant1_save_24['plant'] = '1'
    plant1_save_24['forecast'] = '24'

    plant1_save_48 = plant1_pred_step1.filter(regex='y46')
    plant1_save_48 = plant1_save_48.reset_index()
    plant1_save_48.columns = f_cols
    plant1_save_48['plant'] = '1'
    plant1_save_48['forecast'] = '48'

    plant2_save_24 = plant2_pred_step1.filter(regex='y25')
    plant2_save_24 = plant2_save_24.reset_index()
    plant2_save_24.columns = f_cols
    plant2_save_24['plant'] = '2'
    plant2_save_24['forecast'] = '24'

    plant2_save_48 = plant2_pred_step1.filter(regex='y46')
    plant2_save_48 = plant2_save_48.reset_index()
    plant2_save_48.columns = f_cols
    plant2_save_48['plant'] = '2'
    plant2_save_48['forecast'] = '48'

    for key in test_pred:
        if (key[-3:]) == 'y25':
            if (key[:4]) == 'loc1':
                plant1_save_24['cond_loc1'] = test_pred['loc1_y25'][0]
            if (key[:4]) == 'loc2':
                plant1_save_24['cond_loc2'] = test_pred['loc2_y25'][0]
            if (key[:4]) == 'loc3':
                plant1_save_24['cond_loc3'] = test_pred['loc3_y25'][0]
        if (key[-3:]) == 'y46':
            if (key[:4]) == 'loc1':
                plant1_save_48['cond_loc1'] = test_pred['loc1_y46'][0]
            if (key[:4]) == 'loc2':
                plant1_save_48['cond_loc2'] = test_pred['loc2_y46'][0]
            if (key[:4]) == 'loc3':
                plant1_save_48['cond_loc3'] = test_pred['loc3_y46'][0]

    for key in test2_pred:
        if (key[-3:]) == 'y25':
            if (key[:4]) == 'loc1':
                plant2_save_24['cond_loc1'] = test2_pred['loc1_y25'][0]
            if (key[:4]) == 'loc2':
                plant2_save_24['cond_loc2'] = test2_pred['loc2_y25'][0]
            if (key[:4]) == 'loc3':
                plant2_save_24['cond_loc3'] = test2_pred['loc3_y25'][0]
        if (key[-3:]) == 'y46':
            if (key[:4]) == 'loc1':
                plant2_save_48['cond_loc1'] = test2_pred['loc1_y46'][0]
            if (key[:4]) == 'loc2':
                plant2_save_48['cond_loc2'] = test2_pred['loc2_y46'][0]
            if (key[:4]) == 'loc3':
                plant2_save_48['cond_loc3'] = test2_pred['loc3_y46'][0]

    Prediction.objects.bulk_create(
    Prediction(**vals) for vals in plant1_save_24.to_dict('records')
    )
    Prediction.objects.bulk_create(
    Prediction(**vals) for vals in plant1_save_48.to_dict('records')
    )
    Prediction.objects.bulk_create(
    Prediction(**vals) for vals in plant2_save_24.to_dict('records')
    )
    Prediction.objects.bulk_create(
    Prediction(**vals) for vals in plant2_save_48.to_dict('records')
    )

    # update는 1시간마다 이루어져야 하지만 시뮬레이션을 위해서 임의 시간으로 설정한다.

    # test용 break
    if idx > 30:
        devices = FCMDevice.objects.all()
        devices.send_message("update", "prediction updated")

        while time.time() - start_time < 30:  # 30초
            continue
        if idx > 60:
            break
    # print('Data Updated')





