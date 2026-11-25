"""Pipeline Prefect : orchestration du projet ML de churn."""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task

FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
DATA_PATH = "Churn_Modelling.csv"
MODEL_PATH = "model.joblib"


def run_cmd(cmd):
    """Exécute une commande et lève une erreur si elle échoue."""
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # nosec B603


@task(name="install_dependencies")
def install_task():
    run_cmd([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


@task(name="format_code")
def format_task():
    run_cmd([sys.executable, "-m", "black", *FILES])


@task(name="code_quality")
def quality_task():
    run_cmd([sys.executable, "-m", "pylint", "--fail-under=9", *FILES])


@task(name="code_security")
def security_task():
    run_cmd([sys.executable, "-m", "bandit", "-q", *FILES])


@task(name="unit_tests")
def tests_task():
    run_cmd([sys.executable, "-m", "pytest", "-v", "tests/"])


@task(name="prepare_data")
def prepare_data_task():
    print("Préparation des données...")
    from model_pipeline import prepare_data

    return prepare_data(DATA_PATH)


@task(name="train_model_task")
def train_model_task(data):
    print("Entraînement du modèle...")
    from model_pipeline import train_model

    X_train, X_test, y_train, y_test, preprocessor = data
    model = train_model(X_train, y_train, preprocessor)
    return model, X_test, y_test


@task(name="save_model_task")
def save_model_task(model):
    print("Sauvegarde du modèle...")
    from model_pipeline import save_model

    save_model(model, MODEL_PATH)


@task(name="load_model_task")
def load_model_task():
    print("Chargement du modèle...")
    from model_pipeline import load_model

    return load_model(MODEL_PATH)


@task(name="evaluate_model_task")
def evaluate_model_task(model, X_test, y_test):
    print("Évaluation du modèle...")
    from model_pipeline import evaluate_model

    evaluate_model(model, X_test, y_test)


@task(name="fetch_project")
def fetch_project_task():
    """Récupère la dernière version du projet depuis le repo git."""
    run_cmd(["git", "pull", "origin", "main"])


@task(name="predict_task")
def predict_task(model, X_test):
    """Prédit la classe des 5 premiers clients du jeu de test."""
    print("Prédiction...")
    preds = model.predict(X_test[:5])
    print("Prédictions :", preds)
    return preds


@flow(name="predict")
def predict_flow():
    """Charge le modèle et prédit sur quelques échantillons."""
    data = prepare_data_task()
    _, X_test, _, _, _ = data
    model = load_model_task()
    predict_task(model, X_test)


@flow(name="install")
def install_flow():
    install_task()


@flow(name="code")
def code_flow():
    install_task()
    format_task()
    quality_task()
    security_task()
    tests_task()


@flow(name="all")
def all_flow():
    fetch_project_task()
    install_task()
    format_task()
    quality_task()
    security_task()
    tests_task()
    data = prepare_data_task()
    model, X_test, y_test = train_model_task(data)
    save_model_task(model)
    evaluate_model_task(model, X_test, y_test)


@flow(name="train")
def train_flow():
    data = prepare_data_task()
    train_model_task(data)


@flow(name="evaluate")
def evaluate_flow():
    data = prepare_data_task()
    _, X_test, _, y_test, _ = data
    model = load_model_task()
    evaluate_model_task(model, X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flow",
        choices=["all", "train", "entrainement", "evaluate", "code", "install", "predict"],
        required=True,
    )
    args = parser.parse_args()

    if args.flow == "all":
        all_flow()
    elif args.flow in ("train", "entrainement"):
        train_flow()
    elif args.flow == "evaluate":
        evaluate_flow()
    elif args.flow == "code":
        code_flow()
    elif args.flow == "install":
        install_flow()
    elif args.flow == "predict":
        predict_flow()
