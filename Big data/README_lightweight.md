# Flight Delay Predictor (Lightweight Version)

A web application that simulates airline flight delay predictions without requiring PySpark or a machine learning model.

## Features

- Predicts flight delays based on airport, airline, date, month, and year
- Provides simulated probability of delay
- Identifies likely cause of delay (Carrier, Weather, National Aviation System, or Late Aircraft)
- Shows breakdown of delay cause percentages

## Requirements

- Python 3.8+
- Flask

## Installation

1. Clone this repository

2. Install the required dependency:

```bash
pip install flask
```

3. (Optional) Create the 'models' directory if you want to use the historical estimates file:

```bash
mkdir -p models
```

## Running the Application

1. Start the Flask application:

```bash
python app_lightweight.py
```

2. Open your browser and navigate to http://localhost:5000

## Using the Application

1. Enter the flight details:
   - Select an airport (e.g., ATL, JFK, LAX)
   - Select an airline (e.g., AA, DL, UA)
   - Enter the day (1-31)
   - Select the month (1-12)
   - Enter the year
2. Click "Predict Delay"
3. View the prediction results:
   - Delay status (Expected/Not Expected)
   - Delay probability percentage
   - Most likely cause of delay (if delay is predicted)
   - Breakdown of delay cause percentages

## Prediction Patterns

This lightweight version uses the following patterns to simulate predictions:

- Atlanta (ATL) with Delta (DL) in July: Weather-related delays
- JFK with American Airlines (AA): Carrier-related delays
- Chicago O'Hare (ORD) with United (UA) in winter months: Weather delays
- Denver (DEN) in December/January: Winter weather delays
- Miami (MIA) in summer: Thunderstorm-related delays
- Southwest Airlines (WN): Generally more on-time than other carriers
- All other cases: Random predictions with seasonal factors

## Note

This is a lightweight simulation that doesn't use actual predictive models. If you need accurate delay predictions based on real data, consider using the full version with PySpark when you have more disk space available.

## License

[MIT License](LICENSE) 