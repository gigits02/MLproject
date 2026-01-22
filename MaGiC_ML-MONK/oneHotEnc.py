import numpy as np
import pandas as pd

#Applica l'encoding One-Hot. 
#Esempio:
#Se un attributo può prendere 4 valori interi distinti, converte tutto in binario assegnando:
#1--->1000
#2--->0100
#3--->0010
#4--->0001

#Mappatura del numero di valori possibili per ogni attributo (dataset: MONK)
attribute_sizes = [3, 3, 2, 3, 4, 2]  

def one_hot_encode(row, attribute_sizes):
    encoded = []
    for i, val in enumerate(row):
        one_hot = [0] * attribute_sizes[i]
        one_hot[val - 1] = 1  
        encoded.extend(one_hot)
    return encoded

# Legge il file e lo trasforma in un dataframe
input_file = "./original_MonkFiles/monks-3.test"
output_file = "./encoded_MonkFiles/m3test.csv"

data = pd.read_csv(input_file, delim_whitespace=True, header=None)

# Il target è la prima colonna, gli inputs sono le altre sei
targets = data.iloc[:, 0].values
attributes = data.iloc[:, 1:-1].values

# Applicazione del One-Hot Encoding
encoded_data = [one_hot_encode(row, attribute_sizes) for row in attributes]

# Creazione del nuovo dataframe con i dati trasformati
encoded_df = pd.DataFrame(np.hstack([targets.reshape(-1, 1), encoded_data]))

# Salva il file
encoded_df.to_csv(output_file, index=False, header=False)

print(f"One-Hot Encoding completato! File salvato come: {output_file}")