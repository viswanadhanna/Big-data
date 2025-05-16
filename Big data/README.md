# Flight Delay Predictor

A web application that predicts airline flight delays based on historical data using a machine learning model built with PySpark.

## Features

- Predicts flight delays based on airport, airline, date, month, and year
- Provides probability of delay
- Identifies likely cause of delay (Carrier, Weather, National Aviation System, or Late Aircraft)
- Shows breakdown of delay cause percentages

## Requirements

- Python 3.8+
- Flask
- PySpark 3.3+
- Java 8+ (required for PySpark)
  - On Windows, make sure Java is installed and the JAVA_HOME environment variable is set properly

## Installation

1. Clone this repository

2. Install the required dependencies:

```bash
pip install flask pyspark==3.3.0 boto3
```

3. Create the 'models' directory and 'datasets' directory:

```bash
mkdir -p models datasets
```

4. Download the airline dataset and place it in the datasets folder:
   - Use the "Airline_Delay_Cause.csv" dataset from the Bureau of Transportation Statistics
   - You can use a sample dataset for testing purposes
   - Place the file in the `datasets` directory

## Windows-Specific Setup

1. Install Java 8 or newer from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [OpenJDK](https://adoptopenjdk.net/)

2. Set JAVA_HOME environment variable:
   - Right-click on "This PC" or "My Computer"
   - Click "Properties" > "Advanced system settings" > "Environment Variables"
   - Create a new system variable named JAVA_HOME with the path to your Java installation (e.g., C:\Program Files\Java\jdk1.8.0_xxx)
   - Add %JAVA_HOME%\bin to your PATH variable

3. Install PySpark dependencies:
   - PySpark requires Hadoop winutils.exe for Windows
   - Download winutils.exe for Hadoop from [GitHub](https://github.com/steveloughran/winutils)
   - Set HADOOP_HOME environment variable to the directory containing winutils.exe

## Training the Model

If you want to train the model yourself (optional - sample data is included):

1. Uncomment the training line in main.py:
   - Open main.py and find `# train_model()`
   - Remove the comment character to make it `train_model()`

2. Run the training script:
```bash
python main.py
```

3. This will create model files in the 'models' directory

## Running the Application

1. Start the Flask application:

```bash
python app.py
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

## Troubleshooting

Common issues on Windows:
- Java not found: Make sure JAVA_HOME is set correctly and Java is installed
- PySpark initialization errors: Check that the py4j and other dependencies are correctly installed
- If you have Spark initialization issues, check the logs for specific error messages
- If you get "winutils.exe" errors, make sure HADOOP_HOME is set correctly and winutils.exe is in the bin directory

## Model Training

The application uses a Gradient Boosted Trees Classifier trained on historical airline delay data. The model was trained on features including:

- Year and month (with seasonal transformations)
- Carrier and airport
- Historical flight volume
- Historical delay cause proportions

## Notes

- For demonstration purposes, the application includes a fallback to generate mock predictions if the ML models are not available.
- Special test cases: Try ATL/DL in July or JFK/AA for specific delay predictions.

## License

[MIT License](LICENSE) 