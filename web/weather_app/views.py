from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth import authenticate, login, logout
from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import requests
import matplotlib
from concurrent.futures import ThreadPoolExecutor, as_completed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from .models import WeatherRecord
from .districts import RWANDA_DISTRICTS_COORDS


def is_admin(user):
    return user.is_staff or user.is_superuser


# Normalize predictions so night never shows "sunny"
def normalize_prediction(prediction, time_of_day):
    if not prediction:
        return prediction
    pred = prediction.lower()
    if time_of_day == "night" and pred == "sunny":
        return "clear_night"
    return pred


# -------------------------------
# PROJECT PATH
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import generate_synthetic_weather, load_data
from src.featurize import make_features
from src.model import train_model, save_model, load_model

MODEL_PATH = PROJECT_ROOT / 'model.joblib'
SAMPLE_CSV = PROJECT_ROOT / 'data' / 'sample_weather.csv'


# -------------------------------
# HOME & ADMIN AUTH
# -------------------------------
def home(request):
    return render(request, "weather_app/home.html")


def admin_login_view(request):
    """
    Simple login view specifically for admin/staff users.
    - If already logged in and admin -> go straight to dashboard.
    - Otherwise, show username/password form.
    """
    if request.user.is_authenticated and is_admin(request.user):
        return redirect("admin_dashboard")

    error = None

    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None and is_admin(user):
            login(request, user)
            return redirect("admin_dashboard")
        else:
            error = "Invalid credentials or you are not authorized as admin."

    return render(request, "weather_app/admin_login.html", {"error": error})


@login_required
def admin_logout_view(request):
    """
    Logs out the current user and sends them back to the public home page.
    """
    logout(request)
    return redirect("home")


@login_required
@user_passes_test(is_admin)
def train_view(request):
    # Corrected call
    generate_synthetic_weather(365, seed=42, out_csv=str(SAMPLE_CSV))

    df = load_data(str(SAMPLE_CSV))
    X, y = make_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model, acc, report, _ = train_model(X_train, y_train)
    save_model(model, str(MODEL_PATH))

    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    cm_url = "data:image/png;base64," + base64.b64encode(buf.read()).decode()

    return render(request, "weather_app/train.html", {
        "accuracy": acc,
        "report": report,
        "confusion_matrix_url": cm_url
    })


# -------------------------------
# REAL-TIME WEATHER FETCH BY COORDINATES
# -------------------------------
def fetch_weather_by_coords(district, use_cache=True):
    """
    Fetch weather data for a district with caching support.
    Cache key is based on district name to allow 5-minute caching.
    """
    # Create cache key
    cache_key = f"weather_{district}"
    
    # Try to get from cache first
    if use_cache:
        cached_data = cache.get(cache_key)
        if cached_data:
            return cached_data
    
    coords = RWANDA_DISTRICTS_COORDS.get(district)
    if not coords:
        raise ValueError(f"No coordinates found for {district}")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": coords["lat"],
        "lon": coords["lon"],
        "appid": settings.WEATHER_API_KEY,
        "units": "metric"
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()

    if response.status_code != 200 or "main" not in data:
        raise ValueError(data.get("message", "Weather API error"))

    weather_data = {
        "temp_max": data["main"]["temp_max"],
        "temp_min": data["main"]["temp_min"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "precipitation": data.get("rain", {}).get("1h", 0.0),
        "visibility": data.get("visibility", 10000) / 1000,
        "cloud_cover": data["clouds"]["all"],
        "date": pd.Timestamp.now()
    }
    
    # Cache for 5 minutes (300 seconds)
    if use_cache:
        cache.set(cache_key, weather_data, 300)
    
    return weather_data


def fetch_weather_parallel(districts, max_workers=10):
    """
    Fetch weather data for multiple districts in parallel using ThreadPoolExecutor.
    Returns a dictionary mapping district names to their weather data or errors.
    """
    results = {}
    
    def fetch_single(district):
        try:
            return district, fetch_weather_by_coords(district), None
        except Exception as e:
            return district, None, str(e)
    
    # Use ThreadPoolExecutor for parallel API calls
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_district = {
            executor.submit(fetch_single, district): district 
            for district in districts
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_district):
            district, weather_data, error = future.result()
            results[district] = {
                'weather': weather_data,
                'error': error
            }
    
    return results


def predict_weather_for_district(district, model, time_of_day='afternoon'):
    """Predict weather for a district at a specific time of day.
    
    time_of_day: 'morning' (6 AM), 'afternoon' (2 PM), 'night' (10 PM)
    """
    try:
        weather = fetch_weather_by_coords(district)
        
        # Adjust temperature and other features based on time of day
        hour_map = {'morning': 6, 'afternoon': 14, 'night': 22}
        hour = hour_map.get(time_of_day, 14)
        
        # Create a copy of weather data and adjust for time of day
        weather_data = weather.copy()
        # Ensure date includes hour for proper feature engineering
        now = pd.Timestamp.now()
        weather_data['date'] = pd.Timestamp(
            year=now.year, month=now.month, day=now.day,
            hour=hour, minute=0, second=0, microsecond=0
        )
        
        # Adjust temperature based on time of day (typical diurnal variation)
        temp_adjustment = {'morning': -2, 'afternoon': 0, 'night': -5}
        adjustment = temp_adjustment.get(time_of_day, 0)
        weather_data['temp_max'] = weather_data['temp_max'] + adjustment
        weather_data['temp_min'] = weather_data['temp_min'] + adjustment
        
        # Slight adjustments for other weather parameters based on time
        if time_of_day == 'night':
            # Night typically has lower wind and higher humidity
            weather_data['wind_speed'] = max(0, weather_data['wind_speed'] - 1)
            weather_data['humidity'] = min(100, weather_data['humidity'] + 5)
        elif time_of_day == 'morning':
            # Morning might have slightly higher humidity
            weather_data['humidity'] = min(100, weather_data['humidity'] + 3)
        
        # Create DataFrame and make features
        df = pd.DataFrame([weather_data])
        X, _ = make_features(df)
        raw_prediction = model.predict(X)[0]
        prediction = normalize_prediction(raw_prediction, time_of_day)
        
        return {
            'prediction': prediction,
            'weather': weather_data,
            'time_of_day': time_of_day
        }
    except Exception as e:
        return {
            'prediction': None,
            'weather': None,
            'error': str(e),
            'time_of_day': time_of_day
        }


def predict_view(request):
    if not MODEL_PATH.exists():
        return redirect("train")

    # Try to load model with error handling
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as e:
        # Model loading failed - likely compatibility issue
        # Return error message suggesting retraining
        from django.http import HttpResponse
        error_msg = f"""
        <html>
        <body style="font-family: Arial; padding: 40px;">
            <h2>Model Loading Error</h2>
            <p>The model file cannot be loaded. This is likely due to a compatibility issue with Python 3.13 or scikit-learn version mismatch.</p>
            <p><strong>Error:</strong> {str(e)}</p>
            <p><strong>Solution:</strong> Please retrain the model by visiting <a href="/train/">/train/</a></p>
            <p><a href="/train/" style="background: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Go to Training Page</a></p>
        </body>
        </html>
        """
        return HttpResponse(error_msg)
    
    prediction = None
    weather = None
    error = None
    selected = None
    
    # Get predictions for all districts and all time periods
    # OPTIMIZED: Fetch weather data for all districts in parallel
    all_district_predictions = {}
    
    # Fetch all weather data in parallel (much faster than sequential)
    districts_list = list(RWANDA_DISTRICTS_COORDS.keys())
    weather_results = fetch_weather_parallel(districts_list, max_workers=15)
    
    # Process predictions for each district using batch prediction (more efficient)
    for district in districts_list:
        weather_result = weather_results.get(district, {})
        base_weather = weather_result.get('weather')
        weather_error = weather_result.get('error')
        
        if base_weather:
            try:
                # Use batch prediction for all three time periods at once (more efficient)
                batch_results = batch_predict_weather(model, base_weather, ['morning', 'afternoon', 'night'])
                district_data = {
                    'coords': RWANDA_DISTRICTS_COORDS[district],
                    'morning': batch_results['morning'],
                    'afternoon': batch_results['afternoon'],
                    'night': batch_results['night']
                }
                all_district_predictions[district] = district_data
            except Exception as e:
                # If prediction fails, create empty data structure
                district_data = {
                    'coords': RWANDA_DISTRICTS_COORDS[district],
                    'morning': {'prediction': None, 'weather': None, 'error': str(e)},
                    'afternoon': {'prediction': None, 'weather': None, 'error': str(e)},
                    'night': {'prediction': None, 'weather': None, 'error': str(e)}
                }
                all_district_predictions[district] = district_data
        else:
            # If API call failed, create empty data structure
            district_data = {
                'coords': RWANDA_DISTRICTS_COORDS[district],
                'morning': {'prediction': None, 'weather': None, 'error': weather_error or 'Unknown error'},
                'afternoon': {'prediction': None, 'weather': None, 'error': weather_error or 'Unknown error'},
                'night': {'prediction': None, 'weather': None, 'error': weather_error or 'Unknown error'}
            }
            all_district_predictions[district] = district_data

    if request.method == "POST":
        selected = request.POST.get("district")
        if selected and selected in all_district_predictions:
            # Use afternoon as default for single district view
            result = all_district_predictions[selected]['afternoon']
            prediction = result.get('prediction')
            weather = result.get('weather')
            error = result.get('error')

            # Persist this single-district afternoon prediction to the database
            try:
                if weather and prediction:
                    WeatherRecord.objects.create(
                        user=request.user if request.user.is_authenticated else None,
                        date=weather.get("date", pd.Timestamp.now()).date()
                             if hasattr(weather.get("date", None), "date")
                             else pd.Timestamp.now().date(),
                        temp_max=weather.get("temp_max", 0),
                        temp_min=weather.get("temp_min", 0),
                        humidity=weather.get("humidity", 0),
                        precipitation=weather.get("precipitation", 0),
                        pressure=weather.get("pressure", 1015.0),
                        visibility=weather.get("visibility", 10.0),
                        wind_speed=weather.get("wind_speed", 0),
                        cloud_cover=weather.get("cloud_cover", 0),
                        prediction=prediction,
                    )
            except Exception:
                # If saving fails, ignore so the page still loads
                pass

    from datetime import datetime
    current_date = datetime.now()
    
    return render(request, "weather_app/live_dashboard.html", {
        "districts": list(RWANDA_DISTRICTS_COORDS.keys()),
        "prediction": prediction,
        "weather": weather,
        "error": error,
        "selected": selected,
        "all_district_predictions": all_district_predictions,
        "districts_coords": RWANDA_DISTRICTS_COORDS,
        "current_date": current_date.strftime("%B %d, %Y")
    })


def manual_predict_view(request):
    """
    Allow a user to pick a Rwanda district and manually enter weather values,
    then get a single prediction (plus class probabilities) using predict.html.
    """
    from datetime import datetime

    morning_prediction = None
    afternoon_prediction = None
    night_prediction = None
    error = None
    selected_district = None
    selected_date = None

    # Load model if available
    model = None
    if not MODEL_PATH.exists():
        error = "Model is not trained yet. Please train the model first."
    else:
        try:
            model = load_model(str(MODEL_PATH))
        except Exception as e:
            error = f"Error loading model: {e}"

    # Handle form submit via POST parameters
    if request.method == "POST" and model is not None and not error:
        selected_district = request.POST.get("district") or None
        date_str = request.POST.get("date")

        try:
            if not selected_district:
                raise ValueError("Please select a district.")
            if not date_str:
                raise ValueError("Please provide a date.")

            date_val = pd.to_datetime(date_str)
            selected_date = date_val

            def parse_required(name):
                val = request.POST.get(name)
                if val is None or val == "":
                    raise ValueError(f"{name.replace('_', ' ').title()} is required.")
                return float(val)

            data = {
                "date": date_val,
                "temp_max": parse_required("temp_max"),
                "temp_min": parse_required("temp_min"),
                "precipitation": parse_required("precipitation"),
                "humidity": parse_required("humidity"),
                "wind_speed": parse_required("wind_speed"),
            }

            # Optional numeric fields: default to 0 if empty
            for opt in ["pressure", "visibility", "cloud_cover"]:
                val = request.POST.get(opt)
                data[opt] = float(val) if val not in (None, "") else 0.0

            # Build three rows for morning, afternoon, night with different hours
            rows = []
            hour_map = {"morning": 6, "afternoon": 14, "night": 22}
            temp_adjustment = {"morning": -2, "afternoon": 0, "night": -5}

            for period, hour in hour_map.items():
                row = data.copy()
                # set specific hour on the chosen date
                row["date"] = pd.Timestamp(
                    year=date_val.year,
                    month=date_val.month,
                    day=date_val.day,
                    hour=hour,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                # adjust temperatures similar to live view logic
                adj = temp_adjustment.get(period, 0)
                row["temp_max"] = row["temp_max"] + adj
                row["temp_min"] = row["temp_min"] + adj
                rows.append(row)

            df = pd.DataFrame(rows)
            X, _ = make_features(df)

            # Predict for all three periods
            raw_preds = model.predict(X)
            preds = {
                "morning": normalize_prediction(raw_preds[0], "morning"),
                "afternoon": normalize_prediction(raw_preds[1], "afternoon"),
                "night": normalize_prediction(raw_preds[2], "night"),
            }

            morning_prediction = preds["morning"]
            afternoon_prediction = preds["afternoon"]
            night_prediction = preds["night"]

            # Save main (afternoon) prediction to database for this manual entry
            from .models import WeatherRecord

            try:
                WeatherRecord.objects.create(
                    user=request.user if request.user.is_authenticated else None,
                    date=date_val.date(),
                    temp_max=data["temp_max"],
                    temp_min=data["temp_min"],
                    humidity=data["humidity"],
                    precipitation=data["precipitation"],
                    pressure=data["pressure"],
                    visibility=data["visibility"],
                    wind_speed=data["wind_speed"],
                    cloud_cover=data["cloud_cover"],
                    prediction=afternoon_prediction or "",
                )
            except Exception:
                # Do not break the page if saving fails; just skip persistence
                pass

        except Exception as e:
            error = str(e)

    districts = list(RWANDA_DISTRICTS_COORDS.keys())

    return render(
        request,
        "weather_app/predict.html",
        {
            "districts": districts,
            "selected_district": selected_district,
            "morning_prediction": morning_prediction,
            "afternoon_prediction": afternoon_prediction,
            "night_prediction": night_prediction,
            "selected_date": selected_date,
            "error": error,
        },
    )


def predict_weather_for_district_with_data(district, model, base_weather, time_of_day='afternoon'):
    """Predict weather for a district using existing weather data.
    
    This avoids multiple API calls by reusing the same base weather data.
    """
    try:
        # Adjust temperature and other features based on time of day
        hour_map = {'morning': 6, 'afternoon': 14, 'night': 22}
        hour = hour_map.get(time_of_day, 14)
        
        # Create a copy of weather data and adjust for time of day
        weather_data = base_weather.copy()
        # Ensure date includes hour for proper feature engineering
        now = pd.Timestamp.now()
        weather_data['date'] = pd.Timestamp(
            year=now.year, month=now.month, day=now.day,
            hour=hour, minute=0, second=0, microsecond=0
        )
        
        # Adjust temperature based on time of day (typical diurnal variation)
        temp_adjustment = {'morning': -2, 'afternoon': 0, 'night': -5}
        adjustment = temp_adjustment.get(time_of_day, 0)
        weather_data['temp_max'] = weather_data['temp_max'] + adjustment
        weather_data['temp_min'] = weather_data['temp_min'] + adjustment
        
        # Slight adjustments for other weather parameters based on time
        if time_of_day == 'night':
            # Night typically has lower wind and higher humidity
            weather_data['wind_speed'] = max(0, weather_data['wind_speed'] - 1)
            weather_data['humidity'] = min(100, weather_data['humidity'] + 5)
        elif time_of_day == 'morning':
            # Morning might have slightly higher humidity
            weather_data['humidity'] = min(100, weather_data['humidity'] + 3)
        
        # Create DataFrame and make features
        df = pd.DataFrame([weather_data])
        X, _ = make_features(df)
        raw_prediction = model.predict(X)[0]
        prediction = normalize_prediction(raw_prediction, time_of_day)
        
        return {
            'prediction': prediction,
            'weather': weather_data,
            'time_of_day': time_of_day
        }
    except Exception as e:
        return {
            'prediction': None,
            'weather': None,
            'error': str(e),
            'time_of_day': time_of_day
        }


def batch_predict_weather(model, base_weather, time_periods=['morning', 'afternoon', 'night']):
    """
    Batch predict weather for multiple time periods at once.
    This is more efficient than calling predict_weather_for_district_with_data multiple times.
    """
    try:
        now = pd.Timestamp.now()
        hour_map = {'morning': 6, 'afternoon': 14, 'night': 22}
        temp_adjustment = {'morning': -2, 'afternoon': 0, 'night': -5}
        
        # Prepare data for all time periods
        rows = []
        for time_of_day in time_periods:
            weather_data = base_weather.copy()
            hour = hour_map.get(time_of_day, 14)
            weather_data['date'] = pd.Timestamp(
                year=now.year, month=now.month, day=now.day,
                hour=hour, minute=0, second=0, microsecond=0
            )
            
            adjustment = temp_adjustment.get(time_of_day, 0)
            weather_data['temp_max'] = weather_data['temp_max'] + adjustment
            weather_data['temp_min'] = weather_data['temp_min'] + adjustment
            
            if time_of_day == 'night':
                weather_data['wind_speed'] = max(0, weather_data['wind_speed'] - 1)
                weather_data['humidity'] = min(100, weather_data['humidity'] + 5)
            elif time_of_day == 'morning':
                weather_data['humidity'] = min(100, weather_data['humidity'] + 3)
            
            rows.append(weather_data)
        
        # Batch predict all at once
        df = pd.DataFrame(rows)
        X, _ = make_features(df)
        raw_predictions = model.predict(X)
        
        # Format results
        results = {}
        for i, time_of_day in enumerate(time_periods):
            prediction = normalize_prediction(raw_predictions[i], time_of_day)
            results[time_of_day] = {
                'prediction': prediction,
                'weather': rows[i],
                'time_of_day': time_of_day
            }
        
        return results
    except Exception as e:
        # Return error for all periods if batch fails
        return {
            period: {
                'prediction': None,
                'weather': None,
                'error': str(e),
                'time_of_day': period
            }
            for period in time_periods
        }


# -------------------------------
# ADMIN DASHBOARD
# -------------------------------
@login_required
@user_passes_test(is_admin)
def admin_dashboard_view(request):
    records = WeatherRecord.objects.all().order_by("-date")

    # Labels for all 12 months (used in the stacked monthly overview chart)
    labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Convert to a plain Python list so it renders cleanly into JavaScript
    weather_types = list(
        WeatherRecord.objects.values_list("prediction", flat=True).distinct()
    )

    # Counts per prediction type for each month (12 x N matrix)
    month_counts = []
    for m in range(1, 13):
        month_counts.append([
            WeatherRecord.objects.filter(
                date__month=m, prediction=w
            ).count() for w in weather_types
        ])

    # ------------- Monthly summary (current month only) -------------
    now = datetime.now()
    current_month_index = now.month  # 1-12
    current_month_label = labels[current_month_index - 1]

    current_month_counts = [
        WeatherRecord.objects.filter(
            date__month=current_month_index, prediction=w
        ).count()
        for w in weather_types
    ]
    current_month_total = sum(current_month_counts)

    return render(
        request,
        "weather_app/Weather.html",
        {
            "weather_records": records,
            "month_labels": labels,
            "weather_types": weather_types,
            "month_counts": month_counts,
            "current_month_label": current_month_label,
            "current_month_counts": current_month_counts,
            "current_month_total": current_month_total,
            "current_date": now.strftime("%B %d, %Y"),
        },
    )


@login_required
@user_passes_test(is_admin)
def monthly_summary_view(request):
    """
    Dedicated page that shows aggregated predicted weather data
    for the **current month only**, plus a bar chart summarizing
    how this month stands by prediction type.
    """
    labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    now = datetime.now()
    current_month_index = now.month  # 1-12
    current_month_label = labels[current_month_index - 1]

    weather_types = list(
        WeatherRecord.objects.values_list("prediction", flat=True).distinct()
    )

    # For the current month only, build counts per prediction type
    current_month_counts = [
        WeatherRecord.objects.filter(
            date__month=current_month_index, prediction=w
        ).count()
        for w in weather_types
    ]
    current_month_total = sum(current_month_counts)

    return render(
        request,
        "weather_app/monthly_summary.html",
        {
            "weather_types": weather_types,
            "current_month_label": current_month_label,
            "current_month_counts": current_month_counts,
            "current_month_total": current_month_total,
            "current_date": now.strftime("%B %d, %Y"),
        },
    )


# -------------------------------
# ASYNC API ENDPOINT FOR PROGRESSIVE LOADING
# -------------------------------
def api_predictions(request):
    """
    API endpoint that returns predictions for all districts as JSON.
    This allows the frontend to load predictions asynchronously.
    """
    if not MODEL_PATH.exists():
        return JsonResponse({'error': 'Model not trained'}, status=404)
    
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as e:
        return JsonResponse({'error': f'Model loading error: {str(e)}'}, status=500)
    
    # Fetch all weather data in parallel
    districts_list = list(RWANDA_DISTRICTS_COORDS.keys())
    weather_results = fetch_weather_parallel(districts_list, max_workers=15)
    
    # Build predictions dictionary using batch prediction
    predictions = {}
    for district in districts_list:
        weather_result = weather_results.get(district, {})
        base_weather = weather_result.get('weather')
        weather_error = weather_result.get('error')
        
        if base_weather:
            try:
                # Use batch prediction for efficiency
                batch_results = batch_predict_weather(model, base_weather, ['morning', 'afternoon', 'night'])
                
                # Format for frontend (maintain compatibility with existing format)
                predictions[district] = {
                    'coords': RWANDA_DISTRICTS_COORDS[district],
                    'morning': {
                        'prediction': batch_results['morning'].get('prediction'),
                        'weather': batch_results['morning'].get('weather'),
                        'temp': batch_results['morning'].get('weather', {}).get('temp_max', 0),
                        'humidity': batch_results['morning'].get('weather', {}).get('humidity', 0),
                        'wind': batch_results['morning'].get('weather', {}).get('wind_speed', 0),
                        'cloud': batch_results['morning'].get('weather', {}).get('cloud_cover', 0),
                        'precip': batch_results['morning'].get('weather', {}).get('precipitation', 0),
                    },
                    'afternoon': {
                        'prediction': batch_results['afternoon'].get('prediction'),
                        'weather': batch_results['afternoon'].get('weather'),
                        'temp': batch_results['afternoon'].get('weather', {}).get('temp_max', 0),
                        'humidity': batch_results['afternoon'].get('weather', {}).get('humidity', 0),
                        'wind': batch_results['afternoon'].get('weather', {}).get('wind_speed', 0),
                        'cloud': batch_results['afternoon'].get('weather', {}).get('cloud_cover', 0),
                        'precip': batch_results['afternoon'].get('weather', {}).get('precipitation', 0),
                    },
                    'night': {
                        'prediction': batch_results['night'].get('prediction'),
                        'weather': batch_results['night'].get('weather'),
                        'temp': batch_results['night'].get('weather', {}).get('temp_max', 0),
                        'humidity': batch_results['night'].get('weather', {}).get('humidity', 0),
                        'wind': batch_results['night'].get('weather', {}).get('wind_speed', 0),
                        'cloud': batch_results['night'].get('weather', {}).get('cloud_cover', 0),
                        'precip': batch_results['night'].get('weather', {}).get('precipitation', 0),
                    }
                }
            except Exception as e:
                predictions[district] = {
                    'coords': RWANDA_DISTRICTS_COORDS[district],
                    'error': str(e)
                }
        else:
            predictions[district] = {
                'coords': RWANDA_DISTRICTS_COORDS[district],
                'error': weather_error or 'Unknown error'
            }
    
    return JsonResponse({'predictions': predictions})


# views.py
from django.shortcuts import render


def weather_map(request):
    """
    Lightweight view to render the Rwanda map with district overlays
    without calling the live prediction API. Provides the minimal
    structure that the live_dashboard template expects so labels,
    polygons, and tooltips render correctly.
    """
    # Build placeholder data for every district with coordinates
    # so the template can attach labels and hovers on the GeoJSON.
    def empty_period():
        return {
            "prediction": "",
            "weather": {
                "temp_max": 0,
                "humidity": 0,
                "wind_speed": 0,
                "cloud_cover": 0,
                "precipitation": 0,
            },
        }

    all_district_predictions = {}
    for name, coords in RWANDA_DISTRICTS_COORDS.items():
        all_district_predictions[name] = {
            "coords": coords,
            "morning": empty_period(),
            "afternoon": empty_period(),
            "night": empty_period(),
        }

    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")

    return render(
        request,
        "weather_app/live_dashboard.html",
        {
            "districts": list(RWANDA_DISTRICTS_COORDS.keys()),
            "selected": None,
            "prediction": None,
            "weather": None,
            "error": None,
            "all_district_predictions": all_district_predictions,
            "districts_coords": RWANDA_DISTRICTS_COORDS,
            "current_date": current_date,
        },
    )
