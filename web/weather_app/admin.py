from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from .models import WeatherRecord


# Customize User admin
class MyUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'is_staff', 'is_active', 'date_joined')
    list_filter = ('is_staff', 'is_active')


admin.site.unregister(User)
admin.site.register(User, MyUserAdmin)


# Register WeatherRecord admin
@admin.register(WeatherRecord)
class WeatherRecordAdmin(admin.ModelAdmin):
    list_display = [
        "date", "temp_max", "temp_min", "humidity", "precipitation",
        "pressure", "visibility", "wind_speed", "cloud_cover", "prediction"
    ]
    list_filter = ["date", "temp_max", "temp_min", "humidity"]
    search_fields = ["date"]
