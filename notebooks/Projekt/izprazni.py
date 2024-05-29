import os

def clear_traffic_data1():
    file_path = "data/raw/traffic_data.csv"
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # Izprazni vsebino datoteke
        print(f"Vsebina datoteke {file_path} je bila uspešno izbrisana.")
    except FileNotFoundError:
        print(f"Datoteka {file_path} ne obstaja.")

def clear_traffic_data2():
    file_path = "data/raw/traffic_incidents_data.csv"
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # Izprazni vsebino datoteke
        print(f"Vsebina datoteke {file_path} je bila uspešno izbrisana.")
    except FileNotFoundError:
        print(f"Datoteka {file_path} ne obstaja.")

def clear_traffic_data3():
    file_path = "data/raw/weather_data.csv"
    try:
        with open(file_path, 'w') as file:
            file.truncate(0)  # Izprazni vsebino datoteke
        print(f"Vsebina datoteke {file_path} je bila uspešno izbrisana.")
    except FileNotFoundError:
        print(f"Datoteka {file_path} ne obstaja.")

if __name__ == "__main__":
    clear_traffic_data1()
    clear_traffic_data2()
    clear_traffic_data3()
