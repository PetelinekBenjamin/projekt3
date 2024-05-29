import pandas as pd
import requests

def get_traffic_incidents(api_key, latitude, longitude, radius):
    base_url = "https://api.tomtom.com/traffic/services/5/incidentDetails"
    bbox = f"{longitude - 0.01},{latitude - 0.01},{longitude + 0.01},{latitude + 0.01}"  # Definiramo omejitveni okvir okoli središča
    fields = "{incidents{type,geometry{type,coordinates},properties{iconCategory}}}"
    language = "en-GB"
    t = "1111"
    time_validity_filter = "present"
    url = f"{base_url}?key={api_key}&bbox={bbox}&fields={fields}&language={language}&t={t}&timeValidityFilter={time_validity_filter}"
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            incident_data = response.json()
            return incident_data
        else:
            print("Napaka pri pridobivanju podatkov. Status code:", response.status_code)
            return None
    except Exception as e:
        print("Napaka:", str(e))
        return None

# Nastavitev API ključa
api_key = "pm6qpRlSxt3IDGlZubFpKHtlaWojAqAO"

# Koordinate središča območja, za katerega želite pridobiti podatke o prometu in incidentih
latitude = 40.7128  # Primer: New York City
longitude = -74.0060
radius = 10000  # Polmer območja v metrih

# Pridobitev podatkov o prometu in incidentih
traffic_incidents_data = get_traffic_incidents(api_key, latitude, longitude, radius)

# Preverite, ali so bili podatki pravilno pridobljeni
if traffic_incidents_data:
    # Pretvorimo podatke v DataFrame
    incidents_df = pd.json_normalize(traffic_incidents_data['incidents'])
    incidents_df = pd.DataFrame(traffic_incidents_data['incidents'])

# Štetje števila nesreč
    num_incidents = len(incidents_df)

    print("Število nesreč:", num_incidents)

    df = pd.DataFrame({
        "Stevilo nesrec": [num_incidents]
    })

    df.to_csv(r"data/raw/traffic_incidents_data.csv", index=False)
    print("Podatki o stevilu nesreč uspešno shranjeni")

