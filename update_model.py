import pandas as pd
import numpy as np 
import os
import django

# https://docs.djangoproject.com/en/3.0/topics/settings/#calling-django-setup-is-required-for-standalone-django-usage
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raspberry.settings")
django.setup()

from monitor.models import PlantEnviron, WeatherForecast


path = os.path.abspath(os.path.join(os.path.dirname(__file__), './.data'))

plant1 = pd.read_csv(path + '/plant1_train_split.csv', parse_dates=[0])
plant2 = pd.read_csv(path + '/plant2_train_split.csv', parse_dates=[0])
weather = pd.read_csv(path + '/weather4.csv', parse_dates=[0])

col = [
    'recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2',
    'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3',
    'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3',
]

plant1.columns = col
plant2.columns = col
weather.columns = ['fcTime', 'temp_25', 'temp_46', 'humid_25', 'humid_46', 'rain_25', 'rain_46', 'wind_25', 'wind_46']

plant1.dropna(inplace = True)
plant2.dropna(inplace = True)

PlantEnviron.objects.all().delete()
WeatherForecast.objects.all().delete()

plant1['recTime'] = plant1['recTime'].apply(lambda x : x.to_pydatetime())
plant2['recTime'] = plant2['recTime'].apply(lambda x : x.to_pydatetime())
weather['fcTime'] = weather['fcTime'].apply(lambda x : x.to_pydatetime())

plant1['plant'] = '1'
plant2['plant'] = '2'

PlantEnviron.objects.bulk_create(
    PlantEnviron(**vals) for vals in plant1.to_dict('records')
)

PlantEnviron.objects.bulk_create(
    PlantEnviron(**vals) for vals in plant2.to_dict('records')
)

WeatherForecast.objects.bulk_create(
    WeatherForecast(**vals) for vals in weather.to_dict('records')
)


