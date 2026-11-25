"""Tests de validation du contenu du dataset."""

import pandas as pd
import pytest

DATA_PATH = "Churn_Modelling.csv"
EXPECTED_COLUMNS = [
    "RowNumber", "CustomerId", "Surname", "CreditScore", "Geography",
    "Gender", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA_PATH)


def test_csv_dimensions(df):
    assert df.shape == (10000, 14)


def test_csv_columns(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_duplicate_customers(df):
    assert df["CustomerId"].is_unique


def test_no_missing_values(df):
    assert not df.isna().any().any()


def test_target_values(df):
    assert set(df["Exited"].unique()) == {0, 1}


def test_value_ranges(df):
    assert df["Age"].between(18, 100).all()
    assert df["CreditScore"].between(300, 900).all()
    assert (df["Balance"] >= 0).all()


def test_categories(df):
    assert set(df["Geography"].unique()) == {"France", "Spain", "Germany"}
    assert set(df["Gender"].unique()) == {"Male", "Female"}
