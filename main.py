"""Point d'entrée en ligne de commande du projet ML."""

import argparse

from model_pipeline import (
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
)


def main():

    # 1. Configuration CLI
    parser = argparse.ArgumentParser(description="Machine Learning Churn Prediction")

    parser.add_argument(
        "--data", default="Churn_Modelling.csv", help="Chemin du dataset"
    )

    parser.add_argument("--target", default="Exited", help="Colonne cible")

    parser.add_argument(
        "--model", default="model.joblib", help="Nom du fichier du modele"
    )

    parser.add_argument(
        "--all", action="store_true", help="Executer tout le pipeline Machine Learning"
    )

    # 2. Lecture des arguments
    args = parser.parse_args()

    # 3. Vérification de l'option --all
    if not args.all:
        parser.print_help()
        return

    print("\n========== PIPELINE MACHINE LEARNING ==========\n")

    # 4. Préparation des données
    print("ETAPE 1 : Preparation des donnees")

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        args.data, args.target
    )

    # 5. Entraînement du modèle
    print("\nETAPE 2 : Entrainement du modele")

    model = train_model(X_train, y_train, preprocessor)

    # 6. Évaluation
    print("\nETAPE 3 : Evaluation du modele")

    evaluate_model(model, X_test, y_test)

    # 7. Sauvegarde
    print("\nETAPE 4 : Sauvegarde du modele")

    save_model(model, args.model)

    # 8. Chargement
    print("\nETAPE 5 : Chargement du modele")

    loaded_model = load_model(args.model)

    # 9. Vérification des prédictions
    print("\nETAPE 6 : Verification du modele charge")

    predictions = loaded_model.predict(X_test)

    print("\nPremieres predictions :")
    print(predictions[:10])

    print("\nPipeline termine avec succes !")


if __name__ == "__main__":
    main()
