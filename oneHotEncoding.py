import numpy as np
import pandas as pd

#APPLICA L'ENCODING "ONE-HOT":
#SE UN ATTRIBUTO PUÒ PRENDERE 4 VALORI INTERI DISTINTI, PORTA TUTTO IN BINARIO ASSEGNANDO
#1--->1000
#2--->0100
#3--->0010
#4--->0001
#In questo caso, nel main, si è creato poi il file nel modo appropriato per il problema del MONK (ML project)

#Mappatura del numero di valori per ogni attributo
attribute_sizes = [3, 3, 2, 3, 4, 2]  # Numero di valori distinti per attributo

def one_hot_encode(row, attribute_sizes):
    encoded = []
    for i, val in enumerate(row):
        one_hot = [0] * attribute_sizes[i]
        one_hot[val - 1] = 1  # Attiviamo il bit corrispondente
        encoded.extend(one_hot)
    return encoded

# Legge il file e lo trasforma in un dataframe
input_file = "./monkFiles/monks-3.test"
output_file = "./project/MONK/m3test.csv"

data = pd.read_csv(input_file, delim_whitespace=True, header=None)

# Il target è la prima colonna, gli input sono le altre sei
targets = data.iloc[:, 0].values
attributes = data.iloc[:, 1:-1].values

# Applicazione del One-Hot Encoding
encoded_data = [one_hot_encode(row, attribute_sizes) for row in attributes]

# Creazione del nuovo dataframe con i dati trasformati
encoded_df = pd.DataFrame(np.hstack([targets.reshape(-1, 1), encoded_data]))

# Salva il file
encoded_df.to_csv(output_file, index=False, header=False)

print(f"One-Hot Encoding completato! File salvato come: {output_file}")