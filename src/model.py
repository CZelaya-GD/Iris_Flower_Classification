from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def create_model(input_dim):
    """
    Defines the neural network model.
    """

    model = Sequential([
        Dense(16, activation = "relu", input_dim = input_dim),
        Dropout(0.2),
        Dense(16, activation = "relu"),
        Dropout(0.2),
        Dense(3, activation = "softmax")  # 3 classes for Iris
    ])

    return model 