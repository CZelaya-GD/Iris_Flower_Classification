from src.model import create_model

def test_create_model():
    model = create_model(4)
    assert model is not None
    assert model.input_shape[1] == 4
