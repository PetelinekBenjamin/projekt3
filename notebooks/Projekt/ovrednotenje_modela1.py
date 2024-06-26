import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.compose import ColumnTransformer
import mlflow
import os
from tensorflow.keras.models import load_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
import mlflow.pyfunc
 







# Nastavitev sledenja MLflow
mlflow.set_tracking_uri('https://dagshub.com/PetelinekBenjamin/projekt3.mlflow')
# mlflow.set_tracking_uri("https://dagshub.com/ZanPovseGit/inteligentniSistem.mlflow")

# Nastavitev uporabniškega imena in gesla
os.environ["MLFLOW_TRACKING_USERNAME"] = "PetelinekBenjamin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "30663408e580bdb3f66e074627577a040f36b5ff"
# os.environ["MLFLOW_TRACKING_USERNAME"] = "ZanPovseGit"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "bdf091cc3f58df2c8346bb8ce616545e0e40b351"

def pripravi_podatke_za_ucenje(vrednosti, okno_velikost):
    X, y = [], []
    for i in range(len(vrednosti) - okno_velikost):
        X.append(vrednosti[i:i+okno_velikost, :])
        y.append(vrednosti[i+okno_velikost, -1])
    return np.array(X), np.array(y)

def fill_missing_values(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return X_imputed

# Začetek MLflow teka
with mlflow.start_run(run_name="OvrednotenjeModela1"):


    # Preberi podatke

    pot_do_datoteke = r'data/processed/test_prod.csv'
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

    # Load the scaler from the file
    scaler_path = r"models/model1_scaler_pipeline1.pkl"
    scaler = joblib.load(scaler_path)

    scaler_path1 = r"models/model1_scaler_pipeline2.pkl"
    scaler1 = joblib.load(scaler_path1)

    # Definicija cevovoda za predprocesiranje podatkov
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
        ('scaler', StandardScaler(), ['day',
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
        ('scaler1', StandardScaler(), ['Stevilo nesrec']),
    ])


    processed_data = preprocessing_pipeline.fit_transform(podatki)



    test_data = processed_data
    print("Oblika učnih podatkov:", test_data.shape)


    # Priprava podatkov za model
    okno_velikost = 2
    X_test, y_test = pripravi_podatke_za_ucenje(test_data, okno_velikost)
    y_test = y_test.flatten()



    model_lstm = joblib.load(r"models/model1_model_lstm.pkl")
    
    


    



    



    # Preverjanje uspešnosti modela
    y_pred = model_lstm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Beleženje metrik
    mlflow.log_metrics({"mse": mse, "mae": mae, "r2": r2})

    # Shranjevanje modela v MLflow
    mlflow.keras.log_model(model_lstm, "lstm_model")

    # Prehod trenutnega modela v fazo produkcije
    #mlflow.set_tag("tag", "production1")
    #mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/lstm_model", "OvrednotenjeModela1")


    print("MSE:", mse)
    print("MAE:", mae)
    print("R2:", r2)

    # Pridobite informacije o zadnjem zagonu z oznako "production"
    last_run = mlflow.search_runs(filter_string="tags.tag='production1'",
                              order_by=["start_time DESC"],
                              max_results=1).iloc[0]

    # Pridobite ID zadnjega zagona
    last_run_id = last_run["run_id"]
    print("ID zadnjega zagona s tagom 'production':", last_run_id)


    run_info = mlflow.get_run(last_run_id)
    # Pridobitev metrik
    metrics_production = run_info.data.metrics

    # Pridobitev modela
    # model_uri = "runs:/" + last_run_id + "/lstm_model"
    #model = mlflow.pyfunc.load_model(model_uri)

    #y_pred = model.predict(X_test)
    #mse = mean_squared_error(y_test, y_pred)
    #mae = mean_absolute_error(y_test, y_pred)
    #r2 = r2_score(y_test, y_pred)
    #print("DELA",mse)
    #print("DELA",mae)
    #print("DELA",r2)


    print(metrics_production)

    mse_prod = metrics_production["mse"]
    mae_prod = metrics_production["mae"]
    r2_prod = metrics_production["r2"]

    print("mse_prod", mse_prod)
    print("mae_prod", mae_prod)
    print("r2_prod", r2_prod)

    # Izračun povprečja metrik za testno in produkcijsko okolje
    avg_test = (mse + mae + r2) / 3
    avg_prod = (mse_prod + mae_prod + r2_prod) / 3

    if mse < mse_prod:
        mlflow.set_tag("tag", "production1")
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/lstm_model", "Production1")
    else:
        mlflow.set_tag("tag", "test1")
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/lstm_model", "Test1")


    
   
        