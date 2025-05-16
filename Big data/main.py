# -*- coding: utf-8 -*-
"""
Airline Delay Prediction Model
"""

import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, sum as sum_, lit, coalesce, udf
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import GBTClassifier
from pyspark.sql.types import DoubleType

def initialize_spark():
    """Initialize Spark session for Windows compatibility"""
    # For Windows - let PySpark find Java automatically
    # If needed, uncomment and set JAVA_HOME:
    # os.environ["JAVA_HOME"] = "C:\\Program Files\\Java\\jdk1.8.0_xxx"
    
    spark = SparkSession.builder \
        .appName("AirlineDelayPreprocess") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    
    return spark

def train_model(data_path="datasets/Airline_Delay_Cause.csv"):
    """Train the airline delay prediction model"""
    spark = initialize_spark()
    
    # Load data from local file
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    
    # Clean data
    df = df.dropna(subset=["arr_flights", "arr_del15", "carrier", "airport"]) \
           .filter(col("arr_flights") > 0)
    df = df.withColumn("carrier", coalesce(col("carrier"), lit("Unknown")))
    df = df.withColumn("airport", coalesce(col("airport"), lit("Unknown")))
    print(f"Rows after cleaning: {df.count()}")
    
    # Reduce airport cardinality
    airport_counts = df.groupBy("airport").agg(sum_("arr_flights").alias("total_flights"))
    top_airports = airport_counts.orderBy(col("total_flights").desc()).limit(30).select("airport")
    df = df.join(top_airports.withColumnRenamed("airport", "top_airport"),
                 df.airport == col("top_airport"), "left_outer")
    df = df.withColumn("airport", when(col("top_airport").isNull(), "Other").otherwise(col("airport")))
    df = df.drop("top_airport")
    unique_airports = df.select("airport").distinct().count()
    print(f"Unique airports after grouping: {unique_airports}")
    
    # Calculate delay rate and label
    df = df.withColumn("delay_rate", col("arr_del15") / col("arr_flights"))
    df = df.withColumn("delay_label", when(col("delay_rate") > 0.25, 1).otherwise(0))
    
    # Feature engineering
    df = df.withColumn("month_sin", expr("sin(2 * pi() * month / 12)"))
    df = df.withColumn("month_cos", expr("cos(2 * pi() * month / 12)"))
    df = df.withColumn("carrier_prop", col("carrier_ct") / col("arr_del15"))
    df = df.withColumn("weather_prop", col("weather_ct") / col("arr_del15"))
    df = df.withColumn("nas_prop", col("nas_ct") / col("arr_del15"))
    df = df.withColumn("late_aircraft_prop", col("late_aircraft_ct") / col("arr_del15"))
    df = df.fillna({"carrier_prop": 0, "weather_prop": 0, "nas_prop": 0, "late_aircraft_prop": 0})
    
    # Add interaction term
    df = df.withColumn("carrier_airport_interaction",
                       coalesce(col("carrier").cast("string") + "_" + col("airport"), lit("Unknown_Unknown")))
    
    # Indexing with handleInvalid
    carrier_indexer = StringIndexer(inputCol="carrier", outputCol="carrier_index", handleInvalid="keep")
    airport_indexer = StringIndexer(inputCol="airport", outputCol="airport_index", handleInvalid="keep")
    interaction_indexer = StringIndexer(inputCol="carrier_airport_interaction",
                                        outputCol="interaction_index", handleInvalid="keep")
    carrier_indexer_model = carrier_indexer.fit(df)
    airport_indexer_model = airport_indexer.fit(df)
    interaction_indexer_model = interaction_indexer.fit(df)
    df = carrier_indexer_model.transform(df)
    df = airport_indexer_model.transform(df)
    df = interaction_indexer_model.transform(df)
    
    # Normalize arr_flights
    assembler_temp = VectorAssembler(inputCols=["arr_flights"], outputCol="arr_flights_vec")
    df = assembler_temp.transform(df)
    scaler = StandardScaler(inputCol="arr_flights_vec", outputCol="arr_flights_scaled")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Assemble features
    features = [
        "year", "month_sin", "month_cos", "carrier_index", "airport_index",
        "interaction_index", "arr_flights_scaled", "carrier_prop", "weather_prop",
        "nas_prop", "late_aircraft_prop"
    ]
    assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
    df = assembler.transform(df)
    
    # Handle class imbalance
    class_counts = df.groupBy("delay_label").count().collect()
    total = sum([row["count"] for row in class_counts])
    weights = {row["delay_label"]: total / (2 * row["count"]) for row in class_counts}
    print(f"Class weights: {weights}")
    df = df.withColumn(
        "weight",
        when(col("delay_label") == 0, lit(weights.get(0.0, 1.0)))
        .when(col("delay_label") == 1, lit(weights.get(1.0, 1.0)))
        .otherwise(lit(1.0))
    )
    
    # Split data
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
    
    # Train model (GBTClassifier)
    gbt = GBTClassifier(
        labelCol="delay_label",
        featuresCol="features",
        weightCol="weight",
        maxIter=50,
        maxDepth=10,
        maxBins=100,
        seed=42
    )
    model = gbt.fit(train_df)
    
    # Save model and data locally
    os.makedirs("models", exist_ok=True)
    model.write().overwrite().save("models/gbt_model_improved")
    test_df.write.mode("overwrite").parquet("models/test_data.parquet")
    carrier_indexer_model.write().overwrite().save("models/carrier_indexer")
    airport_indexer_model.write().overwrite().save("models/airport_indexer")
    interaction_indexer_model.write().overwrite().save("models/interaction_indexer")
    scaler_model.write().overwrite().save("models/scaler")
    
    # Compute historical averages for estimation
    historical_estimates = {}
    averages = df.groupBy("airport", "carrier").agg(
        {"arr_flights": "avg", "arr_del15": "avg", "carrier_ct": "avg",
         "weather_ct": "avg", "nas_ct": "avg", "late_aircraft_ct": "avg"}
    ).collect()
    
    for r in averages:
        key = f"{r['airport']}_{r['carrier']}"
        historical_estimates[key] = {
            "arr_flights": float(r["avg(arr_flights)"] or 1000.0),
            "arr_del15": float(r["avg(arr_del15)"] or 100.0),
            "carrier_ct": float(r["avg(carrier_ct)"] or 30.0),
            "weather_ct": float(r["avg(weather_ct)"] or 10.0),
            "nas_ct": float(r["avg(nas_ct)"] or 20.0),
            "late_aircraft_ct": float(r["avg(late_aircraft_ct)"] or 20.0)
        }
    
    # Save historical estimates
    with open("models/historical_estimates.json", "w") as f:
        json.dump(historical_estimates, f, indent=2)
    
    print("Model training and saving complete!")
    spark.stop()

def test_model():
    """Test the model with sample inputs"""
    spark = initialize_spark()
    
    # Function to extract probability
    def extract_prob(prob):
        return float(prob[1])
    extract_prob_udf = udf(extract_prob, DoubleType())
    
    # Load models
    model = GBTClassifier.load("models/gbt_model_improved")
    carrier_indexer_model = StringIndexer.load("models/carrier_indexer")
    airport_indexer_model = StringIndexer.load("models/airport_indexer")
    interaction_indexer_model = StringIndexer.load("models/interaction_indexer")
    scaler_model = StandardScaler.load("models/scaler")
    
    # Load historical estimates
    with open("models/historical_estimates.json", "r") as f:
        historical_estimates = json.load(f)
    
    # User input prediction
    user_inputs = [
        {"airport": "ATL", "airline": "DL", "date": 15, "month": 7, "year": 2025},
        {"airport": "JFK", "airline": "AA", "date": 10, "month": 7, "year": 2025}
    ]
    
    for user_input in user_inputs:
        # Validate inputs
        airport = user_input["airport"]
        carrier = user_input["airline"]
        month = user_input["month"]
        year = user_input["year"]
        
        # Get historical estimates
        key = f"{airport}_{carrier}"
        estimates = historical_estimates.get(key, {
            "arr_flights": 1000.0, "arr_del15": 100.0,
            "carrier_ct": 30.0, "weather_ct": 10.0,
            "nas_ct": 20.0, "late_aircraft_ct": 20.0
        })
        
        # Create DataFrame
        new_data = spark.createDataFrame([{
            "year": year,
            "month": month,
            "carrier": carrier,
            "airport": airport,
            "arr_flights": estimates["arr_flights"],
            "arr_del15": estimates["arr_del15"],
            "carrier_ct": estimates["carrier_ct"],
            "weather_ct": estimates["weather_ct"],
            "nas_ct": estimates["nas_ct"],
            "late_aircraft_ct": estimates["late_aircraft_ct"]
        }])
        
        # Feature engineering
        new_data = new_data.withColumn("month_sin", expr("sin(2 * pi() * month / 12)"))
        new_data = new_data.withColumn("month_cos", expr("cos(2 * pi() * month / 12)"))
        new_data = new_data.withColumn("carrier_prop", col("carrier_ct") / col("arr_del15"))
        new_data = new_data.withColumn("weather_prop", col("weather_ct") / col("arr_del15"))
        new_data = new_data.withColumn("nas_prop", col("nas_ct") / col("arr_del15"))
        new_data = new_data.withColumn("late_aircraft_prop", col("late_aircraft_ct") / col("arr_del15"))
        new_data = new_data.fillna({"carrier_prop": 0, "weather_prop": 0, "nas_prop": 0, "late_aircraft_prop": 0})
        new_data = new_data.withColumn("carrier_airport_interaction",
                                      coalesce(col("carrier").cast("string") + "_" + col("airport"), lit("Unknown_Unknown")))
        
        # Apply indexers
        new_data = carrier_indexer_model.transform(new_data)
        new_data = airport_indexer_model.transform(new_data)
        new_data = interaction_indexer_model.transform(new_data)
        
        # Scale arr_flights
        assembler_temp = VectorAssembler(inputCols=["arr_flights"], outputCol="arr_flights_vec")
        new_data = assembler_temp.transform(new_data)
        new_data = scaler_model.transform(new_data)
        
        # Assemble features
        features = [
            "year", "month_sin", "month_cos", "carrier_index", "airport_index",
            "interaction_index", "arr_flights_scaled", "carrier_prop", "weather_prop",
            "nas_prop", "late_aircraft_prop"
        ]
        assembler = VectorAssembler(inputCols=features, outputCol="features", handleInvalid="skip")
        new_data = assembler.transform(new_data)
        
        # Predict
        predictions = model.transform(new_data)
        predictions = predictions.withColumn("prob_one", extract_prob_udf(col("probability")))
        predictions = predictions.withColumn(
            "calibrated_prediction",
            when(col("prob_one") > 0.65, 1.0).otherwise(0.0)
        )
        
        # Infer delay type
        predictions = predictions.withColumn(
            "likely_cause",
            when(col("calibrated_prediction") == 0, lit("None")).otherwise(
                when((col("carrier_prop") >= col("weather_prop")) &
                     (col("carrier_prop") >= col("nas_prop")) &
                     (col("carrier_prop") >= col("late_aircraft_prop")), lit("Carrier")).otherwise(
                when((col("weather_prop") >= col("nas_prop")) &
                     (col("weather_prop") >= col("late_aircraft_prop")), lit("Weather")).otherwise(
                when(col("nas_prop") >= col("late_aircraft_prop"), lit("NAS")).otherwise(
                lit("Late Aircraft"))))
            )
        )
        
        # Collect and display results
        result = predictions.select(
            "carrier", "airport", "calibrated_prediction", "prob_one", "likely_cause",
            "carrier_prop", "weather_prop", "nas_prop", "late_aircraft_prop"
        ).collect()[0]
        
        print(f"\nPrediction for {result['carrier']} at {result['airport']} in {month}/{year}:")
        print(f"Delay: {'Expected' if result['calibrated_prediction'] == 1 else 'Not Expected'} "
              f"({result['prob_one']*100:.1f}% probability)")
        if result["calibrated_prediction"] == 1:
            print(f"Likely Cause: {result['likely_cause']} "
                  f"(Carrier: {result['carrier_prop']*100:.1f}%, "
                  f"Weather: {result['weather_prop']*100:.1f}%, "
                  f"NAS: {result['nas_prop']*100:.1f}%, "
                  f"Late Aircraft: {result['late_aircraft_prop']*100:.1f}%)")
    
    spark.stop()

if __name__ == "__main__":
    # To train the model, uncomment the line below:
    # train_model()
    
    # To test the model, uncomment the line below:
    # test_model()
    
    # By default, run the Flask app from app.py
    print("Run 'python app.py' to start the web application")

