
from django.urls import path
from . import views

app_name = "monitor"

urlpatterns = [
    path('', views.index, name="index"),
    path('plant/<int:plant>', views.plant, name='plant'),
    path('line/', views.line, name="line"),
]
