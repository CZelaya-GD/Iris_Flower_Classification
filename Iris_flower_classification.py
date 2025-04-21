import os 
from src.data_loading import IrisDataLoader
from src.model import create_model
from src.training import train_model
from src.evaluation import evaluate_model, plot_history

def main(data_path = "iris.csv"):
    """
    Main function to run the Iris classifcation pipeline. 
    """

    # 1. Load and preprocess data
    data_loader = IrisDataLoader(data_path = data_path)
    X_train, X_test, y_train, y_test = data_loader.load_and_split()

    # 2. Create the model 
    input_dim = X_train.shape[1]
    model = create_model(input_dim)

    # 3. Train the model 
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # 4. Evaluate the model    
    evaluate_model(model, X_test, y_test)
    plot_history(history)

if __name__ == "__main__":
    main()