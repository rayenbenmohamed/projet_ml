"""Tests sur l'existence et le contenu des fichiers du projet."""

import re
from pathlib import Path

import joblib
import pytest

ROOT = Path(__file__).resolve().parent.parent
CODE_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]
REQUIRED_FILES = CODE_FILES + [
    "deploiement_prefect.py", "requirements.txt",
    "Churn_Modelling.csv", ".gitignore", "tests/test_pipeline.py",
]


@pytest.mark.parametrize("name", REQUIRED_FILES)
def test_required_file_exists_and_not_empty(name):
    path = ROOT / name
    assert path.exists(), f"{name} manquant"
    assert path.stat().st_size > 0, f"{name} est vide"


@pytest.mark.parametrize("pkg", ["prefect", "black", "bandit", "pytest", "pylint"])
def test_requirements_contains(pkg):
    content = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert pkg in content


@pytest.mark.parametrize("flow", ["all", "train", "evaluate", "code", "install"])
def test_pipeline_defines_flow(flow):
    source = (ROOT / "pipeline_prefect.py").read_text(encoding="utf-8")
    assert f'@flow(name="{flow}")' in source


@pytest.mark.parametrize(
    "task",
    ["install_dependencies", "format_code", "code_quality", "code_security",
     "unit_tests", "prepare_data", "train_model_task", "save_model_task",
     "load_model_task", "evaluate_model_task"],
)
def test_pipeline_defines_task(task):
    source = (ROOT / "pipeline_prefect.py").read_text(encoding="utf-8")
    assert f'@task(name="{task}")' in source


def test_deployment_has_daily_schedule():
    source = (ROOT / "deploiement_prefect.py").read_text(encoding="utf-8")
    assert 'cron="0 2 * * *"' in source


def test_gitignore_excludes_venv():
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "venv/" in content


def test_saved_model_is_loadable():
    path = ROOT / "model.joblib"
    if not path.exists():
        pytest.skip("model.joblib absent : lancer --flow all d'abord")
    assert hasattr(joblib.load(path), "predict")


@pytest.mark.parametrize("name", CODE_FILES + ["deploiement_prefect.py"])
def test_no_secret_in_code(name):
    content = (ROOT / name).read_text(encoding="utf-8")
    assert not re.search(r"ghp_|github_pat_|password\s*=", content)
