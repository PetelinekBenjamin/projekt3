import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from flask import Flask, request, jsonify
import requests
import io
from flask_cors import CORS
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import tensorflow as tf
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
import mlflow



# MongoDB Atlas connection string
MONGO_URI = "mongodb+srv://benjaminpetelinek12871:user123@cluster0.dk6r56e.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Get the database and collection
db = client.metrike  # Replace with your database name
collection1 = db.ovrednotenje_model1 # Replace with your collection name
collection2 = db.ovrednotenje_model2

# Pot do shranjenega modela
model_filename = r"models/model1_model_lstm.pkl"
#model_filename = "models/model_lstm.h5"
# Uvoz modela
with tf.device('/CPU:0'):
    model = joblib.load(model_filename)

# Pot do shranjenega scalerja
scaler_path = r"models/model1_scaler_pipeline1.pkl"
#scaler_path = "models/scaler_pipeline1.pkl"
# Uvoz scalerja
scaler = joblib.load(scaler_path)

# Pot do shranjenega scalerja
scaler_path1 = r"models/model1_scaler_pipeline2.pkl"
#scaler_path1 = "models/scaler_pipeline2.pkl"
# Uvoz scalerja
scaler1 = joblib.load(scaler_path1)









# Pot do shranjenega modela
model_filename2 = r"models/model2_model_lstm.pkl"
#model_filename = "models/model_lstm.h5"
# Uvoz modela
with tf.device('/CPU:0'):
    model2 = joblib.load(model_filename)

# Pot do shranjenega scalerja
scaler_path2 = r"models/model2_scaler_pipeline1.pkl"
#scaler_path = "models/scaler_pipeline1.pkl"
# Uvoz scalerja
scaler2 = joblib.load(scaler_path2)

# Pot do shranjenega scalerja
scaler_path22 = r"models/model2_scaler_pipeline2.pkl"
#scaler_path1 = "models/scaler_pipeline2.pkl"
# Uvoz scalerja
scaler21 = joblib.load(scaler_path22)

app = Flask(__name__)
CORS(app)

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X = []
    for i in range(len(vrednosti) - okno_velikost + 1):
        X.append(vrednosti[i:i+okno_velikost, :])
    return np.array(X)

def fill_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

@app.route('/predict/1', methods=['GET'])
def post_example():
    #pot_do_datoteke = 'data/processed/test_prod.csv'
    pot_do_datoteke = r'data/processed/train_prod.csv'

    df = pd.read_csv(pot_do_datoteke, parse_dates=['Cas'], index_col='Cas')

    # Izloči manjkajoče vrednosti
    print(df.isnull().sum())

    # Sortiranje po času
    df.sort_index(inplace=True)

    # Dodajanje stolpcev day, month in year
    datum = pd.to_datetime(df.index, format='%d/%m/%Y')
    df['day'] = datum.day
    df['month'] = datum.month
    df['year'] = datum.year

    print(df.columns)
    print(df.tail())

    # Filtriranje značilnic
    najdoprinosne_znacilnice = [
    'day',
    'month',
    'year',
    'Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Average speed',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',
    
]
    ciljna_znacilnica = 'Stevilo nesrec'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]



    preprocessing_pipeline = ColumnTransformer([
        ('fill_missing', FunctionTransformer(fill_missing_values), ['Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Average speed',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',
    ]),
        ('scaler', scaler, ['day',
    'month',
    'year',
    'Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Average speed',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',]),
        ('scaler1', scaler1, ['Stevilo nesrec']),
    ])

    processed_data = preprocessing_pipeline.fit_transform(podatki)

    test_data = processed_data
    print("Oblika učnih podatkov:", test_data.shape)

    # Priprava podatkov za model
    okno_velikost = 2
    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]

    y_pred = []
    for i in range(stevilo_podatkov - 7, stevilo_podatkov):
        with tf.device('/CPU:0'):
            pred = model.predict(X_test[i:i+1])
        y_pred.append(pred)

    y_pred_unscaled = scaler1.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    rounded_y_pred = [round(num, 1) for num in y_pred_unscaled.tolist()]

    # Current timestamp
    current_time = datetime.now()

    # Insert each prediction separately with its own timestamp incremented by one hour
    for i, prediction in enumerate(rounded_y_pred):
        prediction_time = current_time + timedelta(hours=i+1)
        prediction_data = {
            "timestamp": prediction_time.isoformat(),
            "prediction": prediction
        }
        try:
            collection1.insert_one(prediction_data)
            print(f"Prediction data for {prediction_time.isoformat()} inserted into MongoDB successfully.")
        except Exception as e:
            print(f"An error occurred while inserting data into MongoDB: {e}")

    return jsonify({"prediction": rounded_y_pred})










@app.route('/predict/2', methods=['GET'])
def post_example2():
    #pot_do_datoteke = 'data/processed/test_prod.csv'
    pot_do_datoteke = r'data/processed/train_prod.csv'

    df = pd.read_csv(pot_do_datoteke, parse_dates=['Cas'], index_col='Cas')

    # Izloči manjkajoče vrednosti
    print(df.isnull().sum())

    # Sortiranje po času
    df.sort_index(inplace=True)

    # Dodajanje stolpcev day, month in year
    datum = pd.to_datetime(df.index, format='%d/%m/%Y')
    df['day'] = datum.day
    df['month'] = datum.month
    df['year'] = datum.year

    print(df.columns)
    print(df.tail())

    # Filtriranje značilnic
    najdoprinosne_znacilnice = [
    'day',
    'month',
    'year',
    'Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Stevilo nesrec',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',
    
]
    ciljna_znacilnica = 'Average speed'
    podatki = df[najdoprinosne_znacilnice + [ciljna_znacilnica]]



    preprocessing_pipeline = ColumnTransformer([
        ('fill_missing', FunctionTransformer(fill_missing_values), ['Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Stevilo nesrec',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',
    ]),
        ('scaler', scaler2, ['day',
    'month',
    'year',
    'Temperatura (2m)',
    'Relativna vlaga (2m)',
    'Temperatura rosisca (2m)',
    'Obcutna temperatura',
    'Verjetnost padavin',
    'Stevilo nesrec',
    'Free flow speed',
    'Current travel time',
    'Free flow travel time',
    'Confidence',]),
        ('scaler1', scaler21, ['Average speed']),
    ])

    processed_data = preprocessing_pipeline.fit_transform(podatki)

    test_data = processed_data
    print("Oblika učnih podatkov:", test_data.shape)

    # Priprava podatkov za model
    okno_velikost = 2
    X_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    stevilo_podatkov = X_test.shape[0]

    y_pred = []
    for i in range(stevilo_podatkov - 7, stevilo_podatkov):
        with tf.device('/CPU:0'):
            pred = model.predict(X_test[i:i+1])
        y_pred.append(pred)

    y_pred_unscaled = scaler1.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    rounded_y_pred = [round(num, 1) for num in y_pred_unscaled.tolist()]

    # Current timestamp
    current_time = datetime.now()

    # Insert each prediction separately with its own timestamp incremented by one hour
    for i, prediction in enumerate(rounded_y_pred):
        prediction_time = current_time + timedelta(hours=i+1)
        prediction_data = {
            "timestamp": prediction_time.isoformat(),
            "prediction": prediction
        }
        try:
            collection2.insert_one(prediction_data)
            print(f"Prediction data for {prediction_time.isoformat()} inserted into MongoDB successfully.")
        except Exception as e:
            print(f"An error occurred while inserting data into MongoDB: {e}")

    return jsonify({"prediction": rounded_y_pred})





@app.route('/ovrednotenje1', methods=['GET'])
def post_example3():
    # Nastavitev Dagshub MLflow sledenja
    mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/projekt3.mlflow')
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"

    predictions = list(collection1.find({}, {'_id': 0, 'timestamp': 1, 'prediction': 1}))

    # Pretvorba časovnih žigov
    for pred in predictions:
        pred['timestamp'] = datetime.fromisoformat(pred['timestamp']).replace(tzinfo=None)

    # Branje CSV datoteke
    csv_path = r'data/processed/train_prod.csv'
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
    latest_metrics = {}
    for metric_name, metric_value in metrics.items():
        latest_metrics[metric_name] = metric_value

    print("Latest metrics:")
    print(latest_metrics)

    return jsonify({'success': True, 'metrics': latest_metrics})


@app.route('/ovrednotenje2', methods=['GET'])
def post_example4():
    # Nastavitev Dagshub MLflow sledenja
    mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/projekt3.mlflow')
    os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"

    predictions = list(collection1.find({}, {'_id': 0, 'timestamp': 1, 'prediction': 1}))

    # Pretvorba časovnih žigov
    for pred in predictions:
        pred['timestamp'] = datetime.fromisoformat(pred['timestamp']).replace(tzinfo=None)

    # Branje CSV datoteke
    csv_path = r'data/processed/train_prod.csv'
    df = pd.read_csv(csv_path, parse_dates=['Cas'])
    df['Cas'] = df['Cas'].dt.tz_localize(None)

    def find_closest(actual_df, prediction_time):
        closest_row = actual_df.iloc[(actual_df['Cas'] - prediction_time).abs().argsort()[:1]]
        return closest_row['Average speed'].values[0]

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
        mlflow.set_tag('ovrednotenje2', 'true')  # Dodajanje taga 'ovrednotenje1'
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
    latest_run = mlflow.search_runs(filter_string="tags.ovrednotenje2 = 'true'", order_by=["start_time DESC"], max_results=1)
    latest_run_id = latest_run.iloc[0].run_id
    metrics = mlflow.get_run(latest_run_id).data.metrics

    # Izpis metrik
    latest_metrics = {}
    for metric_name, metric_value in metrics.items():
        latest_metrics[metric_name] = metric_value

    print("Latest metrics:")
    print(latest_metrics)

    return jsonify({'success': True, 'metrics': latest_metrics})

    





if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0')
