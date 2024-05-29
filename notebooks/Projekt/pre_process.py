import pandas as pd
import os

# Poti do CSV datotek
file_paths = [
    r"data/raw/weather_data.csv",
    r"data/raw/traffic_incidents_data.csv",
    r"data/raw/traffic_data.csv"
]

# Preberi in združi CSV datoteke
dfs = []
for file_path in file_paths:
    if os.path.isfile(file_path):
        dfs.append(pd.read_csv(file_path))

if dfs:
    combined_df = pd.concat(dfs, axis=1)

    output_file = r"data/processed/reference_data.csv"
    
    # Shrani DataFrame v CSV, preveri, če datoteka že obstaja
    write_header = not os.path.exists(output_file)
    
    combined_df.to_csv(output_file, mode='a', index=False, header=write_header)
    print("Podatki uspešno združeni in shranjeni")
else:
    print("Ni podatkov za združevanje.")
