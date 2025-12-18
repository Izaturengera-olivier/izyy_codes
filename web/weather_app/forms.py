from django import forms


class WeatherForm(forms.Form):
    date = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}))
    temp_max = forms.FloatField()
    temp_min = forms.FloatField()
    humidity = forms.FloatField()
    precipitation = forms.FloatField()
    pressure = forms.FloatField()
    visibility = forms.FloatField()
    wind_speed = forms.FloatField()
    cloud_cover = forms.FloatField()
