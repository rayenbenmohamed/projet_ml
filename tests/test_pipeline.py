from model_pipeline import prepare_data

DATA_PATH = "Churn_Modelling.csv"


def test_prepare_data_shapes():
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(DATA_PATH)
    assert X_train.shape[0] == len(y_train)
    assert X_test.shape[0] == len(y_test)
    assert X_train.shape[1] == X_test.shape[1]
    assert preprocessor is not None
