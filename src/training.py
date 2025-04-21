from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping 

def train_model(model, X_train, y_train, X_test, y_test, learning_rate = 0.01, patience = 10, epochs = 200):
    """
    Compiles and trains the model.
    """

    model.compile(
        optimizer = Adam(learning_rate = learning_rate),
        loss = "sparse_categorical_crossentropy",
        metrics = ["sparse_categorical_crossentropy"]
    )

    early_stopping = EarlyStopping(
        monito = "val_loss",
        patience = patience, 
        restore_best_weights = True
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_test, y_test),
        epochs = epochs, 
        callbacks = early_stopping
    )

    return model, history