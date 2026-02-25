#!/usr/bin/env python3

# Importations des modules
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Chemin du dossier processed
processed_dir = Path('data/processed/')

# Récupération des datasets initiaux (créés avec 1_Split.py)
X_train = pd.read_csv(processed_dir / 'X_train.csv')
X_test  = pd.read_csv(processed_dir / 'X_test.csv')

# Creation de copies par sécurité
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Selection des colonnes numériques, afin d'exclure les colonnes dates
cols_numeric = X_train.select_dtypes(include='number').columns

# Normalisation
scaler = StandardScaler()
X_train_scaled[cols_numeric] = scaler.fit_transform(X_train[cols_numeric])
X_test_scaled[cols_numeric]  = scaler.transform(X_test[cols_numeric])

# Exportation des datasets normalisés (dans le même dossier)
X_train_scaled.to_csv(processed_dir / 'X_train_scaled.csv', index=False)
X_test_scaled.to_csv(processed_dir / 'X_test_scaled.csv', index=False)

# Message de confirmation
print("Normalisation effectuée !")
