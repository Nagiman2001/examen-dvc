#!/usr/bin/env python3

# Importations des modules
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


# Chemins des dossiers nécessaires
processed_dir = Path('data/processed/')
models_dir    = Path('models/models/')

# Création du dossier models si inexistant
models_dir.mkdir(parents=True, exist_ok=True)

# Lecture des datasets (avec selection des colonnes numériques)
X_train_scaled = pd.read_csv(processed_dir / 'X_train_scaled.csv')
numeric_cols = X_train_scaled.select_dtypes(include='number').columns
X_train_scaled = X_train_scaled[numeric_cols]

y_train = pd.read_csv(processed_dir / 'y_train.csv')

# Définition des modèles et leurs grilles de paramètres
models = {
    'LinearRegression': {'model': LinearRegression(), 'params': {}},
    'Lasso': {'model': Lasso(max_iter=5000), 'params': {'alpha': [0.01, 0.1, 1.0, 10.0]}},
    'Ridge': {'model': Ridge(), 'params': {'alpha': [0.1, 1.0, 10.0, 50.0, 100.0], 'solver': ['auto', 'svd', 'cholesky', 'lsqr']}},
    'ElasticNet': {'model': ElasticNet(max_iter=5000), 'params': {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.2, 0.5, 0.8]}},
    'RandomForest': {'model': RandomForestRegressor(random_state=42),
         'params': {'n_estimators': [50, 100], 'max_depth': [None, 5, 10], 'min_samples_split': [2, 5]}}
}

# Initialisation des résultats
best_score = -float('inf')
best_model = None
best_model_name = None
best_params = None

# GridSearch !!
for name, model_params in models.items():

    # Récupération du modèle instancié et de ses paramètres respectifs
    model = model_params['model']
    param_grid = model_params['params']

    # Si des paramètes existent !
    if param_grid:
        grid = GridSearchCV(model, param_grid, cv=5, scoring='r2')
        grid.fit(X_train_scaled, y_train.values.ravel())
        score = grid.best_score_
        params = grid.best_params_
        model_to_save = grid.best_estimator_

    # Si pas de paramètres (pour Linear Regression)
    else:
        model.fit(X_train_scaled, y_train.values.ravel())
        score = model.score(X_train_scaled, y_train.values.ravel())
        params = {}
        model_to_save = model

    # Mettre à jour le modèle ayant un score supérieur
    if score > best_score:
        best_score = score
        best_model = model_to_save
        best_model_name = name
        best_params = params

# Sauvegarde du meilleur modèle et des paramètres
joblib.dump(best_model, models_dir / 'best_model.pkl')
joblib.dump(best_params, models_dir / 'best_params.pkl')

print(f"GridSearch effectué !! Score R2 obtenu : {best_score:.4f}")
print(f"Le modèle et les paramètres sont sauvegardés dans {models_dir}")
