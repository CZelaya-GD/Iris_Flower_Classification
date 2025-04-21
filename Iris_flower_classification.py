import os 
from src.data_loading import IrisDataLoader
from src.model import create_model
from src.training import train_model
from src.evaluation import evalute_model, plot_history

def main(data_path = "iris.csv"):
    """
    Main function to run the Iris classifcation pipeline. 
    """

    