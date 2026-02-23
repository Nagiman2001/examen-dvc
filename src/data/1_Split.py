#!/usr/bin/env python3

# Importtations des modules
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# Chemins des fichiers et dossiers
dataset_path = '../../data/raw/raw.csv'     # chemin du dataset d'origine
output_dir = '../../data/processed/'        # dossier de sortie pour les datasets

# Gestion du dossier processed si inexistant
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Récupération du dataset sur les minéraux
df = pd.read_csv(dataset_path)

# Définition des features et la target
X = df.drop(columns=["silica_concentrate"]) # features
y = df["silica_concentrate"]                # Target


# Autre possiblité si on considère la dernière colonne comme variable cible
# dans le cas ou le nom de cette colonne change et si la dernière est bien la targert
# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]


# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Exportation des 4 datasets
X_train.to_csv(f'{output_dir}X_train.csv', index=False)
X_test.to_csv(f'{output_dir}X_test.csv', index=False)
y_train.to_csv(f'{output_dir}y_train.csv', index=False)
y_test.to_csv(f'{output_dir}y_test.csv', index=False)

# Message de confirmation
print("Splitting Data effectuée !")
