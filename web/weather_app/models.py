from django.db import models
from django.contrib.auth.models import User

class WeatherRecord(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    date = models.DateField()
    temp_max = models.FloatField()
    temp_min = models.FloatField()
    humidity = models.FloatField()
    precipitation = models.FloatField()
    pressure = models.FloatField(default=1015.0)
    visibility = models.FloatField(default=10.0)
    wind_speed = models.FloatField()
    cloud_cover = models.FloatField()
    prediction = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']

    def __str__(self):
        return f"{self.date} - {self.prediction}"
