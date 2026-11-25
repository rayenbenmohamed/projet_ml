# ==========================================
# ATELIER 2 : MODULARISATION DU CODE
# Fichier : model_pipeline.py
# ==========================================

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

# ==========================================
# 1. PREPARATION DES DONNEES
# ==========================================


def prepare_data(file_path, target_column="Exited"):
    """
    Charger et préparer le dataset.
    Supprimer les colonnes inutiles.
    Encoder les variables catégorielles.
    Diviser les données en train et test.
    """

    # Charger le fichier CSV
    df = pd.read_csv(file_path)

    print("Dataset chargé avec succès !")
    print("Dimensions :", df.shape)

    # Vérifier la présence de la colonne cible
    if target_column not in df.columns:
        raise ValueError(f"La colonne cible {target_column} est absente du dataset.")

    # Supprimer les colonnes inutiles
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"], errors="ignore")

    # Supprimer les lignes sans valeur cible
    df = df.dropna(subset=[target_column])

    # Séparer les variables indépendantes et la cible
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identifier les colonnes numériques et catégorielles
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Prétraitement des variables numériques
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    # Prétraitement des variables catégorielles
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Combiner les deux prétraitements
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Diviser les données en entraînement et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Préparation des données terminée !")
    print("Taille X_train :", X_train.shape)
    print("Taille X_test :", X_test.shape)

    return X_train, X_test, y_train, y_test, preprocessor


# ==========================================
# 2. ENTRAINEMENT DU MODELE
# ==========================================


def train_model(X_train, y_train, preprocessor):
    """
    Créer et entraîner le modèle Random Forest.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight="balanced"
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    print("Entraînement terminé !")

    return model


# ==========================================
# 3. EVALUATION DU MODELE
# ==========================================


def evaluate_model(model, X_test, y_test):
    """
    Évaluer les performances du modèle.
    """

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n========== EVALUATION ==========")

    print("Accuracy  :", round(accuracy, 4))
    print("Precision :", round(precision, 4))
    print("Recall    :", round(recall, 4))
    print("F1-score  :", round(f1, 4))

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_test, predictions))

    print("\nRapport de classification :")
    print(classification_report(y_test, predictions, zero_division=0))

    return accuracy


# ==========================================
# 4. SAUVEGARDE DU MODELE
# ==========================================


def save_model(model, file_path="model.joblib"):
    """
    Sauvegarder le modèle entraîné avec Joblib.
    """

    joblib.dump(model, file_path)

    print("Modèle sauvegardé dans :", file_path)


# ==========================================
# 5. CHARGEMENT DU MODELE
# ==========================================


def load_model(file_path="model.joblib"):
    """
    Charger un modèle sauvegardé.
    """

    model = joblib.load(file_path)

    print("Modèle chargé avec succès !")

    return model
