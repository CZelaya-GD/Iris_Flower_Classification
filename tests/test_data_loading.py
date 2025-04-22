from src.data_loading import load_and_preprocess

def test_load_and_preprocess():
    X_train, X_test, y_train, y_test = load_and_preprocess('iris.csv')
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
