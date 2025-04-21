import pandas as pd 
from sklearn.model_selection import train_test_split

class IrisDataLoader:
    """
    Loads and preprocess the Iris dataset.
    """

    def __init__(self, data_path, test_size=0.2, random_state=42):

        self.data_path = data_path 
        self.test_size = test_size
        self.random_state = random_state
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_split(self):
        """
        Loads data, preprocesses, and splits into training/testing sets.
        """

        self.df = pd.read_csv(self.data_path)
        X = self.df.drop("species", axis = 1)
        y = self.df["species"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = self.test_size, random_state = self.random_state)

        return self.X_train, self.X_test, self.y_train, self.y_test