def kopiraj_csv(originalna_pot, nova_pot):
    try:
        # Preveri, ali je datoteka CSV
        if not originalna_pot.endswith('.csv'):
            print("Napaka: Podana datoteka ni CSV datoteka.")
            return
        
        # Preberi vsebino originalne datoteke
        with open(originalna_pot, 'r') as f:
            vsebina = f.read()
        
        # Zapiši vsebino v novo datoteko
        with open(nova_pot, 'w') as f:
            f.write(vsebina)
        
        print(f"Datoteka uspešno prekopirana na novo lokacijo: {nova_pot}")
    except Exception as e:
        print(f"Napaka: {e}")

# Originalna pot do CSV datoteke
originalna_pot = 'data/processed/reference_data.csv'

# Nova pot, kamor želiš kopirati datoteko
nova_pot = 'data/processed/current_data.csv'

# Klic funkcije za kopiranje datoteke
kopiraj_csv(originalna_pot, nova_pot)
