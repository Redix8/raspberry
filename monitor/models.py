from django.db import models


class PlantEnviron(models.Model):
    PLANT_CHOICE = [('1', '1'), ('2', '2')]
    plant         = models.CharField(max_length=1, choices=PLANT_CHOICE)
    recTime       = models.DateTimeField()
    tem_in_loc1	  = models.FloatField()
    hum_in_loc1	  = models.FloatField()
    tem_coil_loc1 = models.FloatField()
    tem_in_loc2	  = models.FloatField()
    hum_in_loc2	  = models.FloatField()
    tem_coil_loc2 = models.FloatField()	
    tem_in_loc3	  = models.FloatField()  
    hum_in_loc3	  = models.FloatField()  
    tem_coil_loc3 = models.FloatField()	    
    tem_out_loc1  = models.FloatField()	
    hum_out_loc1  = models.FloatField()   
    cond_loc1	  = models.BooleanField()
    cond_loc2	  = models.BooleanField()
    cond_loc3     = models.BooleanField()


class WeatherForecast(models.Model):
    fcTime   = models.DateTimeField()
    temp_25	 = models.FloatField() 
    temp_46	 = models.FloatField()      
    humid_25 = models.FloatField()      	
    humid_46 = models.FloatField()      	
    rain_25	 = models.FloatField()      
    rain_46	 = models.FloatField()      
    wind_25	 = models.FloatField()      
    wind_46  = models.FloatField()   


class Prediction(models.Model):
    PLANT_CHOICE = [('1', '1'), ('2', '2')]
    FORECAST_CHOICE = [('24', '24'), ('48', '48')]
    plant         = models.CharField(max_length=1, choices=PLANT_CHOICE)
    forecast      = models.CharField(max_length=2, choices=FORECAST_CHOICE)
    recTime       = models.DateTimeField()
    tem_in_loc1	  = models.FloatField()
    hum_in_loc1	  = models.FloatField()
    tem_coil_loc1 = models.FloatField()
    tem_in_loc2	  = models.FloatField()
    hum_in_loc2	  = models.FloatField()
    tem_coil_loc2 = models.FloatField()	
    tem_in_loc3	  = models.FloatField()  
    hum_in_loc3	  = models.FloatField()  
    tem_coil_loc3 = models.FloatField()	    
    tem_out_loc1  = models.FloatField()	
    hum_out_loc1  = models.FloatField()
    cond_loc1     = models.BooleanField()
    cond_loc2     = models.BooleanField()
    cond_loc3     = models.BooleanField()

    
