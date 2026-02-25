#!/usr/bin/env python3

# Importation des modules
import numpy as np
import pandas as pd
import joblib
import json

from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Chemins des dossiers nécessaires
processed_dir = Path('data/processed')  # Pour récupérer X_test_scaled.csv et y_test.csv
models_dir    = Path('models')    	# Pour récupérer le modèle final et enregistrer predictions.csv
metrics_dir   = Path('metrics')

# Création des dossiers metrics et models/data/ si inexistants
metrics_dir.mkdir(parents=True, exist_ok=True)
(models_dir / 'data').mkdir(parents=True, exist_ok=True)

# Lecture des datasets Test et chargement du modèle final
X_test_scaled = pd.read_csv(processed_dir / 'X_test_scaled.csv')
numeric_cols = X_test_scaled.select_dtypes(include='number').columns
X_test_scaled = X_test_scaled[numeric_cols]

y_test = pd.read_csv(processed_dir / 'y_test.csv').values.ravel()
final_model = joblib.load(models_dir / 'models' / 'final_model.pkl')

# On fait la Prédiction
y_pred = final_model.predict(X_test_scaled)

# Nouveau dataset contenant les prédictions
predictions = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})
predictions.to_csv(models_dir / 'data' / 'predictions.csv', index=False)

# Calcul des métriques liées à la Régression en comparant les valeurs réelles et prédites
# Coefficient de détermination pour la variance des données, mesure si le modèle est bon ou pas en général
r2  = r2_score(y_test, y_pred)
# erreur moyenne du modèle
mae = mean_absolute_error(y_test, y_pred)
# erreur moyenne du modèle, pénalise les grosses erreurs
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)

# Création d'un dictionnaire des métriques
scores = {
    'R2': r2,
    'mse' : mse,
    'RMSE': rmse,
    'MAE': mae
}

# Sauvegarde des métriques
scores_file = metrics_dir / 'scores.json'
with open(scores_file, 'w') as f:
    json.dump(scores, f, indent=4)

print("Évaluate effectuée ! Scores sauvegardés dans", scores_file)
