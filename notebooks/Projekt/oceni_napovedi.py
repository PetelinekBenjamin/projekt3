import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import mlflow
import os

# MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://benjaminpetelinek12871:user123@cluster0.dk6r56e.mongodb.net/?retryWrites=true&w=majority"

# Nastavitev Dagshub MLflow sledenja
mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/projekt3.mlflow')
os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"

# povezava s strežnikom
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client.metrike
collection = db.ovrednotenje_model1

# Pridobivanje vseh napovedi
predictions = list(collection.find({}, {'_id': 0, 'timestamp': 1, 'prediction': 1}))

# Pretvorba časovnih žigov
for pred in predictions:
    pred['timestamp'] = datetime.fromisoformat(pred['timestamp']).replace(tzinfo=None)

# Branje CSV datoteke
csv_path = r'C:\Users\benja\Desktop\JOKE\Projekt-master\Projekt-master\data\processed\train_prod.csv'
df = pd.read_csv(csv_path, parse_dates=['Cas'])


df['Cas'] = df['Cas'].dt.tz_localize(None)


def find_closest(actual_df, prediction_time):
    closest_row = actual_df.iloc[(actual_df['Cas'] - prediction_time).abs().argsort()[:1]]
    return closest_row['Stevilo nesrec'].values[0]

# Izračun napak napovedi
errors = []

for pred in predictions:
    pred_time = pred['timestamp']
    pred_value = pred['prediction']
    actual_value = find_closest(df, pred_time)
    error = abs(pred_value - actual_value)
    errors.append(error)
    print(f"Prediction time: {pred_time}, Prediction value: {pred_value}, Actual value: {actual_value}, Error: {error}")

# Izračun povprečne napake
mean_error = np.mean(errors)
print(f"Mean prediction error: {mean_error}")

# Izračun dodatnih metrik
mae = np.mean(errors)
mse = np.mean(np.square(errors))
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# MLflow
with mlflow.start_run():
    mlflow.set_tag('ovrednotenje1', 'true')  # Dodajanje taga 'ovrednotenje1'
    
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("MSE", mse)
    mlflow.log_metric("RMSE", rmse)

# shranimo napake in metrike v CSV
error_df = pd.DataFrame({
    'timestamp': [pred['timestamp'] for pred in predictions],
    'prediction': [pred['prediction'] for pred in predictions],
    'error': errors
})

metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE'],
    'Value': [mae, mse, rmse]
})

error_df.to_csv('prediction_errors.csv', index=False)
metrics_df.to_csv('prediction_metrics.csv', index=False)


# Preberite metrike iz najnovejšega teka z tagom 'ovrednotenje1'
latest_run = mlflow.search_runs(filter_string="tags.ovrednotenje1 = 'true'", order_by=["start_time DESC"], max_results=1)
latest_run_id = latest_run.iloc[0].run_id

metrics = mlflow.get_run(latest_run_id).data.metrics

# Izpis metrik
print("Latest metrics:")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value}")