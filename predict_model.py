import os
import pickle
import numpy as np
import pandas as pd
from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
import django
from django.conf import settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raspberry.settings")
django.setup()

from monitor.models import PlantEnviron, WeatherForecast, Prediction

def dewpoint(temp, humid):
    return ((243.12 *((17.62 * temp /(243.12 + temp)) + np.log(humid / 100.0))) 
            / (17.62-((17.62 * temp / (243.12 + temp)) + np.log(humid/ 100.0))))

plant1 = PlantEnviron.objects.filter(plant='1')
plant2 = PlantEnviron.objects.filter(plant='2')
forecast = WeatherForecast.objects.all()

col = ['recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2', 
 'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3', 
 'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3', ]

fc_col = ['fcTime', 'temp_25', 'temp_46', 'humid_25', 'humid_46', 'rain_25', 'rain_46', 'wind_25', 'wind_46']

df1 = pd.DataFrame(columns=col)
df2 = pd.DataFrame(columns=col)
fc = pd.DataFrame(columns=fc_col)

for obj in plant1[:31]:
    data = pd.Series({'recTime':obj.recTime, 'tem_in_loc1':obj.tem_in_loc1, 'hum_in_loc1':obj.hum_in_loc1, 'tem_coil_loc1':obj.tem_coil_loc1,
    'tem_in_loc2':obj.tem_in_loc2, 'hum_in_loc2':obj.hum_in_loc2, 'tem_coil_loc2':obj.tem_coil_loc2, 
    'tem_in_loc3':obj.tem_in_loc3, 'hum_in_loc3':obj.hum_in_loc3, 'tem_coil_loc3':obj.tem_coil_loc3,
    'tem_out_loc1':obj.tem_out_loc1, 'hum_out_loc1':obj.hum_out_loc1,
    'cond_loc1':obj.cond_loc1, 'cond_loc2':obj.cond_loc2, 'cond_loc3':obj.cond_loc3,})
    df1 = df1.append(data, ignore_index=True)
df1.index = df1['recTime']
df1 = df1.drop('recTime', axis=1)

for obj in plant2[:31]:
    data = pd.Series({'recTime':obj.recTime, 'tem_in_loc1':obj.tem_in_loc1, 'hum_in_loc1':obj.hum_in_loc1, 'tem_coil_loc1':obj.tem_coil_loc1,
    'tem_in_loc2':obj.tem_in_loc2, 'hum_in_loc2':obj.hum_in_loc2, 'tem_coil_loc2':obj.tem_coil_loc2, 
    'tem_in_loc3':obj.tem_in_loc3, 'hum_in_loc3':obj.hum_in_loc3, 'tem_coil_loc3':obj.tem_coil_loc3,
    'tem_out_loc1':obj.tem_out_loc1, 'hum_out_loc1':obj.hum_out_loc1,
    'cond_loc1':obj.cond_loc1, 'cond_loc2':obj.cond_loc2, 'cond_loc3':obj.cond_loc3,})
    df2 = df2.append(data, ignore_index=True)
df2.index = df2['recTime']
df2 = df2.drop('recTime', axis=1)

for obj in forecast:
    data = pd.Series({'fcTime':obj.fcTime, 'temp_25':obj.temp_25, 'temp_46':obj.temp_46,
    'humid_25':obj.humid_25, 'humid_46':obj.humid_46, 'rain_25':obj.rain_25, 'rain_46':obj.rain_46,
    'wind_25':obj.wind_25, 'wind_46':obj.wind_46})
    fc = fc.append(data, ignore_index=True)
fc.index = fc['fcTime']
fc = fc.drop('fcTime', axis=1)

data  = pd.concat([df1,fc], axis=1)
data2 = pd.concat([df2,fc], axis=1)

# 1공장
# 보간
inp = data.loc[:, 'temp_25':'wind_46']
data.update(inp.interpolate())
tempTrain = data.dropna()

test = tempTrain.resample('1h').asfreq().dropna()
test_X = test.drop(['cond_loc1', 'cond_loc2', 'cond_loc3'], axis=1)
ma6 =  test_X.rolling(6).mean().filter(regex='(25|46)').add_prefix('MA6_')
ma24 = test_X.rolling(24).mean().filter(regex='(25|46)').add_prefix('MA24_')
test_X = pd.concat([test_X, ma6, ma24], axis=1).dropna()

# 2공장
# 보간
inp = data2.loc[:, 'temp_25':'wind_46']
data2.update(inp.interpolate())
tempTrain2 = data2.dropna()

test2 = tempTrain2.resample('1h').asfreq().dropna()
test2_X = test2.drop(['cond_loc1', 'cond_loc2', 'cond_loc3'], axis=1)
ma6 =  test2_X.rolling(6).mean().filter(regex='(25|46)').add_prefix('MA6_')
ma24 = test2_X.rolling(24).mean().filter(regex='(25|46)').add_prefix('MA24_')
test2_X = pd.concat([test2_X, ma6, ma24], axis=1).dropna()

# 모델 불러오기
with open('models/forecast_model.bin','rb') as f:
    forecasters = pickle.load(f)

with open('models/scalers.bin','rb') as f:
    scalers = pickle.load(f)

with open('models/classifiers.bin','rb') as f:
    classifiers = pickle.load(f)

# plant1 predict
plant1_pred = pd.DataFrame()
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
    plant1_pred[col] = pred  
plant1_pred.index = test_X.index

# plant2 predict
plant2_pred = pd.DataFrame()
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
    plant2_pred[col] = pred  
plant2_pred.index = test_X.index




# 1공장 결로 예측 전처리
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

# 2공장 결로 예측 전처리
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

print(plant1_pred)

###################################################################################################

# 1공장 결로 예측
test_pred = {}
for time_label in ['y25', 'y46']:
    X_time = plant1_pred.filter(regex=f'{time_label}')
    for loc_label in ['loc1', 'loc2', 'loc3']:
        print(f'pred : {loc_label}_{time_label}')
        in_col = X_time.filter(regex=f'(in|coil)_{loc_label}').columns.to_list()
        out_col = X_time.filter(regex=f'out_loc1').columns.to_list()
        date_col = ['month','day', 'hour']
        tcol = in_col + out_col + date_col

        p = np.zeros(X_time.shape[0])
        for m in classifiers['1'][f'{time_label}_{loc_label}']:
            p += (m.predict_proba( plant1_pred[tcol] )/5)[:, 1].reshape(-1,)[0]
            p_cond = np.where(p>0.3, 1, 0)
        test_pred[f'{loc_label}_{time_label}'] = p_cond
print(test_pred)

# 2공장 결로 예측
test2_pred = {}
for time_label in ['y25', 'y46']:
    X_time = plant2_pred.filter(regex=f'{time_label}')
    for loc_label in ['loc1', 'loc2', 'loc3']:
        print(f'pred : {loc_label}_{time_label}')
        in_col = X_time.filter(regex=f'(in|coil)_{loc_label}').columns.to_list()
        out_col = X_time.filter(regex=f'out_loc1').columns.to_list()
        date_col = ['month','day', 'hour']
        tcol = in_col + out_col + date_col

        p = np.zeros(X_time.shape[0])
        for m in classifiers['1'][f'{time_label}_{loc_label}']:
            p += (m.predict_proba( plant2_pred[tcol] )/5)[:, 1].reshape(-1,)[0]
            p_cond = np.where(p>0.3, 1, 0)
        test2_pred[f'{loc_label}_{time_label}'] = p_cond



# #save env data
# Prediction.objects.bulk_create(
#     Prediction(**vals) for vals in plant1_pred_25.to_dict('records')
# )
