import requests
import pandas as pd

def get_traffic_data(api_key, latitude, longitude, radius):
    url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": api_key,
        "point": f"{latitude},{longitude}",
        "radius": radius
    }
    
    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            traffic_data = response.json()
            return traffic_data
        else:
            print("Napaka pri pridobivanju podatkov. Status code:", response.status_code)
            return None
    except Exception as e:
        print("Napaka:", str(e))
        return None

# Nastavitev API ključa
api_key = "pm6qpRlSxt3IDGlZubFpKHtlaWojAqAO"

# Koordinate središča območja, za katerega želite pridobiti podatke o prometu
latitude = 40.7128  # Primer: New York City
longitude = -74.0060
radius = 10000  # Polmer območja v metrih

# Pridobitev podatkov o prometu
traffic_data = get_traffic_data(api_key, latitude, longitude, radius)

# Preverite, ali so bili podatki pravilno pridobljeni
if traffic_data:
    # Iz podatkov o prometu izluščimo povprečno hitrost
    average_speed = traffic_data['flowSegmentData']['currentSpeed']
    
    # Dodamo dodatne podatke o prometu
    flow_segment_data = traffic_data.get('flowSegmentData', {})
    frc = flow_segment_data.get('frc', None)
    free_flow_speed = flow_segment_data.get('freeFlowSpeed', None)
    current_travel_time = flow_segment_data.get('currentTravelTime', None)
    free_flow_travel_time = flow_segment_data.get('freeFlowTravelTime', None)
    confidence = flow_segment_data.get('confidence', None)
    road_closure = flow_segment_data.get('roadClosure', None)
    
    # Ustvari DataFrame
    df = pd.DataFrame({
        "Average speed": [average_speed],
        "FRC": [frc],
        "Free flow speed": [free_flow_speed],
        "Current travel time": [current_travel_time],
        "Free flow travel time": [free_flow_travel_time],
        "Confidence": [confidence],
        "Road closure": [road_closure]
    })

    # Shrani DataFrame v CSV
    df.to_csv(r"data/raw/traffic_data.csv", index=False)
    print("Podatki o prometu so bili uspešno shranjeni v 'traffic_data.csv'.")
