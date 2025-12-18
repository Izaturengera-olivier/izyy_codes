from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('admin-login/', views.admin_login_view, name='admin_login'),
    path('admin-logout/', views.admin_logout_view, name='admin_logout'),
    path('train/', views.train_view, name='train'),
    path('predict/', views.predict_view, name='predict'),
    path('manual-predict/', views.manual_predict_view, name='manual_predict'),
    path('dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('monthly-summary/', views.monthly_summary_view, name='monthly_summary'),
    path('map/', views.weather_map, name='weather_map'),
    path('api/predictions/', views.api_predictions, name='api_predictions'),
]
