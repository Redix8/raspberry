import pandas as pd
import numpy as np 
import os
import django

# https://docs.djangoproject.com/en/3.0/topics/settings/#calling-django-setup-is-required-for-standalone-django-usage
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "raspberry.settings")
django.setup()

from monitor.models import PlantOneEnviron, PlantTwoEnviron, WeatherForecast


path = os.path.abspath(os.path.join(os.path.dirname(__file__), './.data'))
print(path)

plant1 = pd.read_csv(path + '/plant1_train.csv', parse_dates=[0])
plant2 = pd.read_csv(path + '/plant2_train.csv', parse_dates=[0])
weather = pd.read_csv(path + '/weather4.csv', parse_dates=[0])

col = ['recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2', 
 'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3', 
 'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3', ]

plant1.columns = col
plant2.columns = col
weather.columns = ['fcTime', 'temp_25', 'temp_46', 'humid_25', 'humid_46', 'rain_25', 'rain_46', 'wind_25', 'wind_46']


# PlantOneEnviron.objects.all().delete()

# for i in range(len(plant1)):
#     data = plant1.iloc[i, :].to_dict()
#     data['recTime'] = data['recTime'].to_pydatetime()
#     PlantOneEnviron(**data).save()        
    
# PlantTwoEnviron.objects.all().delete()

# for i in range(len(plant2)):
#     data = plant2.iloc[i, :].to_dict()
#     data['recTime'] = data['recTime'].to_pydatetime()
#     PlantTwoEnviron(**data).save()       


# for i in range(len(weather)):
#     data = weather.iloc[i, :].to_dict()
#     data['fcTime'] = data['fcTime'].to_pydatetime()
#     WeatherForecast(**data).save()    

