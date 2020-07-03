
from django.urls import path
from . import views

app_name = "monitor"

urlpatterns = [
    path('', views.index, name="index"),
    path('plant1/', views.plant1, name='plant1'),
    path('line/', views.line, name="line"),
]
