import pandas as pd

def sort_csv_by_validation_loss(input_file, output_file):
    # Leggere il file CSV
    df = pd.read_csv(input_file)

    # Estrarre il valore numerico dalla colonna "Validation Loss" (prima di "±")
    df["Validation Loss Value"] = df["Validation Loss"].str.split("±").str[0].astype(float)

    # Ordinare in base al valore della validation loss
    df_sorted = df.sort_values(by="Validation Loss Value", ascending=True)

    # Rimuovere la colonna temporanea
    df_sorted = df_sorted.drop(columns=["Validation Loss Value"])

    # Salvare il file ordinato
    df_sorted.to_csv(output_file, index=False)

    print(f"File ordinato salvato in: {output_file}")

# Esempio di utilizzo
input_file = "./GridSearch/Tanh/MSEgridSearch.csv"  # Modifica con il percorso corretto
output_file = "./GridSearch/Tanh/MSEsorted.csv"
sort_csv_by_validation_loss(input_file, output_file)
