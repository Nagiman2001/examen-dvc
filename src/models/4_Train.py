#!/usr/bin/env python3

# Importation des modules
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# Chemins des dossiers nécessaires
processed_dir = Path('data/processed')
models_dir    = Path('models/models')

# Lecture des datasets normalisés (en excluant les colonnes non-numériques)
X_train_scaled = pd.read_csv(processed_dir / 'X_train_scaled.csv')
numeric_cols = X_train_scaled.select_dtypes(include='number').columns
X_train_scaled = X_train_scaled[numeric_cols]

y_train = pd.read_csv(processed_dir / 'y_train.csv')

# Chargement du meilleur modèles et ses paramètres
best_params = joblib.load(models_dir / 'best_params.pkl')
best_model = joblib.load(models_dir / 'best_model.pkl')

# Récupération de la classe du meilleur modèle
model_class = type(best_model)

# Instanciation PROPRE du meilleur modèle et de ses paramètres
# Si paramètres existants : Ridge, Lasso, ElasticNet ou RandomForest
if best_params:  #
    final_model = model_class(**best_params)
# Si paramètres inexistants : LinearRegression
else:
    final_model = model_class()

# Re-Entraînement Train
final_model.fit(X_train_scaled, y_train.values.ravel())

# Sauvegarde du modèle final
joblib.dump(final_model, models_dir / 'final_model.pkl')

print(f"Train effectué ! Modèle final sauvegardé dans {models_dir / 'final_model.pkl'}")
