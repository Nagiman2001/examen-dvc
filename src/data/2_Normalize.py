#!/usr/bin/env python3

# Importations des modules
import pandas as pd

from sklearn.preprocessing import StandardScaler

# Chemin du dossier processed
processed_dir = '../../data/processed/'

# Récupération des datasets initiaux (créés avec 1_Split.py)
X_train = pd.read_csv(f'{processed_dir}X_train.csv')
X_test  = pd.read_csv(f'{processed_dir}X_test.csv')

# Normalisation
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Exportation des datasets normalisés (dans le même dossier)
X_train_scaled.to_csv(f'{processed_dir}X_train_scaled.csv', index=False)
X_test_scaled.to_csv(f'{processed_dir}X_test_scaled.csv', index=False)

# Message de confirmation
print("Normalisation effectuée !")
