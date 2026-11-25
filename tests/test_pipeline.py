"""Tests unitaires du pipeline ML."""

import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)

DATA_PATH = "Churn_Modelling.csv"


@pytest.fixture(scope="module")
def data():
    return prepare_data(DATA_PATH)


@pytest.fixture(scope="module")
def model(data):
    x_train, _, y_train, _, preprocessor = data
    return train_model(x_train, y_train, preprocessor)


def test_prepare_data_shapes(data):
    x_train, x_test, y_train, y_test, preprocessor = data
    assert x_train.shape[0] == len(y_train)
    assert x_test.shape[0] == len(y_test)
    assert x_train.shape[1] == x_test.shape[1]
    assert preprocessor is not None


def test_split_ratio(data):
    x_train, x_test, _, _, _ = data
    assert len(x_train) + len(x_test) == 10000
    assert len(x_test) == 2000


def test_no_missing_after_prepare(data):
    x_train, x_test, _, _, _ = data
    assert not pd.DataFrame(x_train).isna().any().any()
    assert not pd.DataFrame(x_test).isna().any().any()


def test_missing_file():
    with pytest.raises(Exception):
        prepare_data("fichier_inexistant.csv")


def test_predictions_are_binary(model, data):
    _, x_test, _, y_test, _ = data
    preds = model.predict(x_test)
    assert len(preds) == len(y_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_accuracy_threshold(model, data):
    _, x_test, _, y_test, _ = data
    assert evaluate_model(model, x_test, y_test) >= 0.80


def test_save_and_load(model, data, tmp_path):
    _, x_test, _, _, _ = data
    path = str(tmp_path / "m.joblib")
    save_model(model, path)
    assert np.array_equal(model.predict(x_test), load_model(path).predict(x_test))
