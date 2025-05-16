# BigData

✈️ Flight Delay Prediction with PySpark

This repository contains a full pipeline for predicting flight delays using PySpark, including classification and regression models. The project uses historical flight, airline, and airport data to predict both departure and arrival delays.

📂 Project Structure

predict_flights.txt       # Main PySpark script
models/                   # Directory to save trained models

🚀 Features

Data sampling and enrichment via joins (airlines & airports)
Null handling and outlier treatment
Feature engineering:
Delay ratio
Haversine distance
Time-based features (hour, day, month)
Categorical encoding with OneHotEncoder
Distance bucketing and standard scaling

ML Pipelines with:
Logistic Regression (classification)
Random Forest Regressor (regression)

Evaluation:
AUC for classification
RMSE for regression
Delay propagation patterns
Model saving and prediction on new samples

📊 Example Output
AUC for departure delay classification
RMSE for arrival delay prediction
Correlation between actual and predicted delays
Example delay predictions for 5 test flights

📦 Requirements
Apache Spark (tested with PySpark 3.x)
Hadoop YARN environment
Python 3.x
Datasets for flights, airlines, and airports

🧪 How to Run
Set up your Spark and Hadoop environment.
Replace "path" in the script with actual dataset paths.

Run the script:
spark-submit predict_flights.txt
The script is configured to use 4GB memory for both driver and executors, and assumes YARN as the cluster manager.

💾 Model Output
The trained models are saved under:
models/departure_delay_model/
models/arrival_delay_model/
These can be reloaded for batch inference or integration with real-time pipelines.

🔮 Predict on New Data
You can use the predict_delays(new_data) function defined in the script to run predictions on unseen flights (as a Spark DataFrame).

📌 Notes
Sampling is used to speed up processing (25% of flights).
Outlier handling uses IQR and bounding.
Categorical nulls are replaced with "UNKNOWN".

📬 Contact
For any questions or suggestions, feel free to open an issue or pull request.
