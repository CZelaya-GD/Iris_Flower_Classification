# Iris_Flower_Classification

![scikit-learn](https://img.shields.io/badge/scikit_learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![tensorflow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)

## Overview

This project demonstrates a machine learning workflow to classify iris flowers into three species—*Iris setosa*, *Iris versicolor*, and *Iris virginica*—based on sepal and petal measurements. The codebase is organized for clarity, reproducibility, and ease of extension, following best practices in code modularity, testing, and automation.

---

## Dataset

The dataset used is the classic [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), containing 150 samples with four features each:
- Sepal length
- Sepal width
- Petal length
- Petal width

Each sample is labeled with its species. The dataset is included as `iris.csv` in this repository.

---

## Project Structure

iris-classification/
├── src/
│ ├── data_loading.py # Data loading and preprocessing
│ ├── model.py # Model architecture
│ ├── training.py # Training logic
│ ├── evaluation.py # Evaluation and visualization
├── tests/
│ ├── test_data_loading.py # Unit tests for data loading
│ ├── test_model.py # Unit tests for model creation
├── iris.csv # Dataset
├── main.py # Main pipeline entry point
├── requirements.txt # Dependencies
├── README.md # Project documentation
└── .github/
└── workflows/
└── ci.yml # GitHub Actions CI pipeline

text

---

## Quickstart

1. **Clone the repository:**
git clone https://github.com/yourusername/iris-classification.git
cd iris-classification

text

2. **Install dependencies:**
pip install -r requirements.txt

text

3. **Run the main pipeline:**
python main.py

text

4. **Run tests:**
pytest tests/

text

---

## Features

- **Clean, modular code** using OOP principles.
- **Automated testing** with `pytest`.
- **Continuous Integration (CI/CD)** via GitHub Actions.
- **Reproducible results** and easy experiment tracking.
- **Clear documentation** and extensible structure.

---

## Model and Results

The project uses a simple neural network with two hidden layers and dropout for regularization. After training, the model typically achieves >95% accuracy on the test set. Training and validation curves are plotted for further analysis.

---

## Continuous Integration

All pushes and pull requests to the `main` branch trigger automated tests via GitHub Actions. The workflow is defined in `.github/workflows/ci.yml` and ensures code quality and reproducibility.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features. All code should include appropriate tests and documentation.

---

## License

MIT

---

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) for the Iris dataset.
- Open-source contributors and the Python ML community.
