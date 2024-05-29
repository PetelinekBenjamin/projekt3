import requests
import json
import datetime
import pandas as pd

# URL za pridobivanje podatkov o vremenu
url = "https://api.open-meteo.com/v1/forecast"

# Določitev trenutnega časa
current_time = datetime.datetime.now()
print("datetime now: ",current_time)

# Priprava parametrov za zahtevo
params = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "hourly": ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation_probability"],
    "timezone": "Europe/London",
    "past_days": 0,  # Zdaj pridobiva podatke samo za trenutni dan
    "forecast_days": 1  # Ne pridobi napovedi za prihodnje dni
}

# Pridobivanje podatkov
response = requests.get(url, params=params)

# Preverjanje zahtevka
if response.status_code == 200:
    # Pretvorba odgovora v JSON format
    data = response.json()

    # Izberi najbližji čas v seznamu
    closest_time = min(data["hourly"]["time"], key=lambda x: abs(datetime.datetime.fromisoformat(x) - current_time))
    print("najbljizji cas: ",closest_time)

    # Indeks najbližjega časa
    closest_index = data["hourly"]["time"].index(closest_time)

    # Izberi podatke za najbližji čas
    temperature = data["hourly"]["temperature_2m"][closest_index]
    humidity = data["hourly"]["relative_humidity_2m"][closest_index]
    dew_point = data["hourly"]["dew_point_2m"][closest_index]
    apparent_temperature = data["hourly"]["apparent_temperature"][closest_index]
    precipitation_probability = data["hourly"]["precipitation_probability"][closest_index]

    # Ustvari DataFrame
    df = pd.DataFrame({
        "Cas": [closest_time],
        "Temperatura (2m)": [temperature],
        "Relativna vlaga (2m)": [humidity],
        "Temperatura rosisca (2m)": [dew_point],
        "Obcutna temperatura": [apparent_temperature],
        "Verjetnost padavin": [precipitation_probability]
    })

    # Shrani DataFrame v CSV
    df.to_csv(r"data/raw/weather_data.csv", index=False)
    print("Podatki so bili uspešno shranjeni v 'weather_data.csv'.")
    print(df)
else:
    print("Napaka pri pridobivanju podatkov o vremenu.")
