import matplotlib.pyplot as plt 

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and plots results.
    """

    loss, accuracy = model.evaluate(X_test , y_test, vrbose = 0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

def plot_history(history):
    """
    Plots training histry (loss and accuracy).
    """

    plt.figure(figsize = (12, 4))

    plt.subplot(1,2,1)
    plt.plot(history.history["loss"], 
             label = "Training Loss")
    plt.plot(history.history["val_loss"], 
             label = "Validation Loss")
    plt.legend()
    plt.title("Loss Over Epochs")

    plt.subplot(1,2,3)
    plt.plot(history.history["spase_categorical_accuracy"],
             label = "Training Accuracy")
    plt.plot(history.history["val_sparse_categorical_accuracy"], 
             label = "Validation Accuracy")
    plt.legend()
    plt.title("Accuracy Over Epochs")

    plt.show()