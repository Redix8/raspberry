from django.contrib import admin
from .models import PlantEnviron, WeatherForecast, Prediction


class PlantEnvironAdmin(admin.ModelAdmin):
    list_display = (
        'plant', 'recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2',
        'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3', 
        'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3', 
    )


class WeatherForecastAdmin(admin.ModelAdmin):
    list_display = (
        'fcTime', 'temp_25', 'temp_46', 'humid_25', 'humid_46', 'rain_25', 'rain_46', 'wind_25', 'wind_46', 
    ) 

class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        'plant','forecast', 'recTime', 'tem_in_loc1', 'hum_in_loc1', 'tem_coil_loc1', 'tem_in_loc2',
        'hum_in_loc2', 'tem_coil_loc2', 'tem_in_loc3', 'hum_in_loc3', 'tem_coil_loc3', 
        'tem_out_loc1', 'hum_out_loc1', 'cond_loc1', 'cond_loc2', 'cond_loc3', 
    )

admin.site.register(PlantEnviron, PlantEnvironAdmin)
admin.site.register(WeatherForecast, WeatherForecastAdmin)
admin.site.register(Prediction, PredictionAdmin)
