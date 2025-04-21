from src.data_loading import IrisDataLoader

def test_load_and_split():

    loader = IrisDataLoader(data_path = "iris.csv")
    X_train, X_test, y_train, y_test = loader.load_and_split()
    assert len(X_train) > 0
    assert len(X_test) > 0