import requests

INDIAN_CITIES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867}
}

def fetch_realtime_air_quality_batch():
    """
    Fetches real-time air quality for all major Indian cities in a single fast API call.
    """
    lats = [c["lat"] for c in INDIAN_CITIES.values()]
    lons = [c["lon"] for c in INDIAN_CITIES.values()]
    
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lats,
        "longitude": lons,
        "current": ["pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone"],
        "timezone": "auto"
    }
    
    try:
        response = requests.get(url, params=params)
        results = {}
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                for i, (city_name, coords) in enumerate(INDIAN_CITIES.items()):
                    current = data[i].get("current", {})
                    results[city_name] = {
                        "lat": coords["lat"],
                        "lon": coords["lon"],
                        "pm10": current.get("pm10", 0),
                        "pm2_5": current.get("pm2_5", 0),
                        "carbon_monoxide": current.get("carbon_monoxide", 0),
                        "nitrogen_dioxide": current.get("nitrogen_dioxide", 0),
                        "ozone": current.get("ozone", 0)
                    }
            return results
    except Exception as e:
        print(f"Error fetching data: {e}")
    return {}
