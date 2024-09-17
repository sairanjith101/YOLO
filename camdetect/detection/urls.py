from django.urls import path
from detection import views

urlpatterns = [
    path('', views.index, name='index'),
    path('start_detection/', views.start_detection, name='start_detection'),
    path('stop_detection/', views.stop_detection, name='stop_detection'),
    path('video_feed/', views.video_feed, name='video_feed'),
]