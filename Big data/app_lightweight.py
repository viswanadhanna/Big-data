import os
import json
import random
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# Load historical estimates if available
HISTORICAL_ESTIMATES = {}
try:
    if os.path.exists("models/historical_estimates.json"):
        with open("models/historical_estimates.json", "r") as f:
            HISTORICAL_ESTIMATES = json.load(f)
except Exception:
    pass

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Validate inputs
    required_fields = ['airport', 'airline', 'day', 'month', 'year']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400
    
    airport = data['airport']
    carrier = data['airline']
    month = int(data['month'])
    year = int(data['year'])
    day = int(data['day'])
    
    # Generate prediction using mock data logic
    return generate_prediction(airport, carrier, month, day, year)

def generate_prediction(airport, carrier, month, day, year):
    """Generate prediction data for demo purposes"""
    
    # Special cases for demonstration
    if airport == 'ATL' and carrier == 'DL' and month == 7:
        prediction = {
            "isDelayed": True,
            "probability": 0.78,
            "cause": "Weather",
            "causeBreakdown": {
                "carrier": 0.15,
                "weather": 0.65,
                "nas": 0.12,
                "lateAircraft": 0.08
            }
        }
    elif airport == 'JFK' and carrier == 'AA':
        prediction = {
            "isDelayed": True,
            "probability": 0.85,
            "cause": "Carrier",
            "causeBreakdown": {
                "carrier": 0.70,
                "weather": 0.10,
                "nas": 0.15,
                "lateAircraft": 0.05
            }
        }
    elif airport == 'ORD' and carrier == 'UA' and (month == 12 or month == 1 or month == 2):
        # Chicago winter weather delays
        prediction = {
            "isDelayed": True,
            "probability": 0.88,
            "cause": "Weather",
            "causeBreakdown": {
                "carrier": 0.10,
                "weather": 0.75,
                "nas": 0.10,
                "lateAircraft": 0.05
            }
        }
    elif airport == 'DEN' and (month == 12 or month == 1):
        # Denver winter delays
        prediction = {
            "isDelayed": True,
            "probability": 0.72,
            "cause": "Weather",
            "causeBreakdown": {
                "carrier": 0.15,
                "weather": 0.70,
                "nas": 0.05,
                "lateAircraft": 0.10
            }
        }
    elif airport == 'MIA' and (month >= 6 and month <= 9):
        # Miami summer thunderstorm season
        prediction = {
            "isDelayed": day % 2 == 0,  # Variation based on day
            "probability": 0.65,
            "cause": "Weather",
            "causeBreakdown": {
                "carrier": 0.20,
                "weather": 0.60,
                "nas": 0.10,
                "lateAircraft": 0.10
            }
        }
    elif carrier == 'WN':
        # Southwest tends to be on time more often
        random_prob = random.random() * 0.6  # Lower probability of delay
        is_delayed = random_prob > 0.4
        
        prediction = {
            "isDelayed": is_delayed,
            "probability": random_prob,
            "cause": "Carrier" if is_delayed else "None",
            "causeBreakdown": {
                "carrier": 0.50,
                "weather": 0.20,
                "nas": 0.20,
                "lateAircraft": 0.10
            }
        }
    else:
        # Look up historical data if available
        key = f"{airport}_{carrier}"
        if key in HISTORICAL_ESTIMATES:
            # Use historical data to influence randomness
            estimates = HISTORICAL_ESTIMATES[key]
            delay_rate = estimates.get("arr_del15", 100) / estimates.get("arr_flights", 1000)
            random_prob = random.random() * 0.7 + (delay_rate * 0.3)
        else:
            # Completely random if no historical data
            random_prob = random.random()
        
        is_delayed = random_prob > 0.5
        
        # Generate random cause breakdown with some logic
        if month in [12, 1, 2]:  # Winter
            weather_factor = 0.5
        elif month in [6, 7, 8]:  # Summer
            weather_factor = 0.3
        else:
            weather_factor = 0.2
            
        carrier_pct = random.random() * 0.7
        weather_pct = random.random() * weather_factor
        nas_pct = random.random() * 0.4
        late_pct = random.random() * 0.3
        
        # Normalize
        total = carrier_pct + weather_pct + nas_pct + late_pct
        carrier_pct /= total
        weather_pct /= total
        nas_pct /= total
        late_pct /= total
        
        # Determine main cause
        max_pct = max(carrier_pct, weather_pct, nas_pct, late_pct)
        if max_pct == carrier_pct:
            cause = "Carrier"
        elif max_pct == weather_pct:
            cause = "Weather"
        elif max_pct == nas_pct:
            cause = "NAS"
        else:
            cause = "Late Aircraft"
        
        prediction = {
            "isDelayed": is_delayed,
            "probability": random_prob,
            "cause": cause if is_delayed else "None",
            "causeBreakdown": {
                "carrier": carrier_pct,
                "weather": weather_pct,
                "nas": nas_pct,
                "lateAircraft": late_pct
            }
        }
    
    return jsonify(prediction)

if __name__ == '__main__':
    print("Starting lightweight version of Flight Delay Predictor...")
    print("Note: This version uses simulated data only and does not require PySpark.")
    app.run(debug=True, host='127.0.0.1', port=5000) 