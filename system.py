import os
import pickle
import numpy as np
import pandas as pd
import django
import time
from django_pandas.io import read_frame

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raspberry.settings")
django.setup()

from monitor.models import PlantEnviron, WeatherForecast, Prediction


with open('models/forecast_model.bin', 'rb') as f:
    forecasters = pickle.load(f)

with open('models/scalers.bin', 'rb') as f:
    scalers = pickle.load(f)

plant1_new = pd.read_csv('.data/plant1_test_split.csv', parse_dates=[0])
plant2_new = pd.read_csv('.data/plant2_test_split.csv', parse_dates=[0])
forecast = read_frame(WeatherForecast.objects.all())

for i in range(len(plant1_new)):
    start_time = time.time()
    # 새로 들어온 데이터
    new1 = plant1_new.iloc[i, :]
    new2 = plant2_new.iloc[i, :]

    # 새 데이터 db 저장

    # 날씨 데이터( 전처리 했다고 가정하고 DB에서 불러온다)

    # 1차 예측을 위해서 데이터 처리(feature 생성)

    # 1차 예측

    # 1차 예측으로 2차 예측 시작

    # 2차 예측을 위해서 데이터 처리(feature 생성)

    # 2차 예측 시행

    print('Predict Done')
    # 1차, 2차 예측 DB 저장

    # update는 1시간마다 이루어져야 하지만 시뮬레이션을 위해서 2분으로 설정한다.
    while time.time() - start_time < 2*60:
        continue
    print('Data Updated')



