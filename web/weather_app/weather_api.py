import requests

API_KEY = "PUT_YOUR_OPENWEATHER_API_KEY_HERE"

def fetch_live_weather(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY,
        "units": "metric"
    }

    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    return {
        "temp_max": data["main"]["temp"],
        "temp_min": data["main"]["temp"] - 2,
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "cloud_cover": data["clouds"]["all"],
        "precipitation": data.get("rain", {}).get("1h", 0),
        "visibility": data.get("visibility", 10000) / 1000
    }
