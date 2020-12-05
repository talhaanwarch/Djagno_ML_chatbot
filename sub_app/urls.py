from django.urls import path
from . import views

urlpatterns = [
    path('question/', views.call_model.as_view())
    ]