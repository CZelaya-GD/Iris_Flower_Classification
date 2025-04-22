from src.data_loading import load_and_preprocess
from src.model import create_model
from src.training import train_model
from src.evaluation import evaluate_model, plot_history

def main():
    
    X_train, X_test, y_train, y_test = load_and_preprocess('iris.csv')
    model = create_model(X_train.shape[1])
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    plot_history(history)

if __name__ == "__main__":
    main()
