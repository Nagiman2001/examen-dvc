#!/bin/bash

# Split des données
dvc stage add -n split \
    -d src/data/1_Split.py \
    -d data/raw/raw.csv \
    -o data/processed/X_train.csv \
    -o data/processed/X_test.csv \
    -o data/processed/y_train.csv \
    -o data/processed/y_test.csv \
    python src/data/1_Split.py

# Normalisation
dvc stage add -n normalize \
    -d src/data/2_Normalize.py \
    -d data/processed/X_train.csv \
    -d data/processed/X_test.csv \
    -o data/processed/X_train_scaled.csv \
    -o data/processed/X_test_scaled.csv \
    python src/data/2_Normalize.py

# GridSearch
dvc stage add -n gridsearch \
    -d src/models/3_GridSearch.py \
    -d data/processed/X_train_scaled.csv \
    -d data/processed/y_train.csv \
    -o models/data/best_model.pkl \
    -o models/data/best_params.pkl \
    python src/models/3_GridSearch.py

# Entraînement du modèle
dvc stage add -n train \
    -d src/models/4_Train.py \
    -d data/processed/X_train_scaled.csv \
    -d data/processed/y_train.csv \
    -d models/data/best_model.pkl \
    -d models/data/best_params.pkl \
    -o models/data/final_model.pkl \
    python src/models/4_Train.py

# Évaluation du modèle
dvc stage add -n evaluate \
    -d src/models/5_Evaluate.py \
    -d data/processed/X_test_scaled.csv \
    -d data/processed/y_test.csv \
    -d models/data/final_model.pkl \
    -o models/data/predictions.csv \
    -o metrics/scores.json \
    python src/models/5_Evaluate.py

