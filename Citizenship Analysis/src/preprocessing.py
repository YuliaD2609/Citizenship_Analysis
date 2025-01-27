import os

import pandas.core.frame
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer

def load_and_clean_data(file_path):
    # Carica i dati dal file Excel
    data_cit_d01 = pd.read_excel(file_path, sheet_name='Data - Cit_D01', skiprows=1)
    data_cit_d02 = pd.read_excel(file_path, sheet_name='Data - Cit_D02', skiprows=2)
    print(f"Dataset letto")

    data_cit_d01['Year'] = pd.to_numeric(data_cit_d01['Year'], errors='coerce')
    data_cit_d02['Year'] = pd.to_numeric(data_cit_d02['Year'], errors='coerce')
    year_min = 2014
    regions_to_keep = ["EU 14", "EU 2", "EU 8", "EU Other", "Europe Other"]
    data_cit_d01 = filter_data_by_year_and_region(data_cit_d01, year_min, regions_to_keep)
    data_cit_d02 = filter_data_by_year_and_region(data_cit_d02, year_min, regions_to_keep)
    print(f"Dataset filtrato")

    # Unisci i dataset
    merged_data = pd.merge(data_cit_d01, data_cit_d02, on=['Year', 'Nationality'], how='outer')

    # Seleziona le colonne rilevanti
    selected_columns = ['Application type group_x', 'Nationality', 'Applications',
       'Application type group_y', 'Application type', 'Sex', 'Age', 'Grants']

    #istogrammi per vedere quante cittadinanze sono state concesse a seconda dei valori definiti
    for col in ['Nationality', 'Application type', 'Sex', 'Age']:
        plot_data(merged_data, col, 'citizenship-datasets')

    data = merged_data[selected_columns]
    print(f"Dimensioni del dataset combinato: {data.shape}")
    print(f"Colonne del dataset combinato: {data.columns}")

    # Imputazione dei valori mancanti
    imputer = SimpleImputer(strategy='most_frequent')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Codifica delle variabili categoriche
    data_encoded = pd.get_dummies(data_imputed, columns=['Application type group_x', 'Nationality', 'Applications',
       'Application type group_y', 'Application type', 'Sex', 'Age'], drop_first=True)

    # Separazione delle caratteristiche e del target
    X = data_encoded.drop('Grants', axis=1)
    y = data_encoded['Grants']
    print(f"Dataset processato")

    return X, y

def filter_data_by_year_and_region(df, year, regions):
    """
    Filtra un DataFrame per un anno minimo e una lista di regioni specificate.

    Args:
        df (pd.DataFrame): Il DataFrame da filtrare.
        year (int): L'anno minimo.
        regions (list): La lista delle regioni da includere.

    Returns:
        pd.DataFrame: Il DataFrame filtrato.
    """
    return df[(df['Year'] >= year) & (df['Region'].isin(regions))]

def plot_data(data_to_plot: pandas.core.frame.DataFrame, Column: str, name_data: str) -> None:

    # Cartella per salvare i grafici (creata se non esiste)
    save_path = os.path.join('../Data/Plots', name_data)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10, 8))
    sns.histplot(data=data_to_plot[Column], kde=True)
    plt.title(f"Citizenships granted by {Column}")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{Column}_histplot.png'))
    plt.close()
    print(f"Istogramma per {Column} salvato in {save_path}")

    return
