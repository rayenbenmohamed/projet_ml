"""Tests d'intégration : chaîne complète et interface en ligne de commande."""

import subprocess
import sys

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def test_full_pipeline_end_to_end(tmp_path):
    x_train, x_test, y_train, y_test, prep = prepare_data("Churn_Modelling.csv")
    model = train_model(x_train, y_train, prep)
    path = str(tmp_path / "e2e.joblib")
    save_model(model, path)
    accuracy = evaluate_model(load_model(path), x_test, y_test)
    assert accuracy >= 0.80


def test_cli_rejects_unknown_flow():
    result = subprocess.run(
        [sys.executable, "pipeline_prefect.py", "--flow", "inconnu"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_cli_help_lists_flows():
    result = subprocess.run(
        [sys.executable, "pipeline_prefect.py", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0
    for flow in ["all", "train", "evaluate", "code", "install"]:
        assert flow in result.stdout
