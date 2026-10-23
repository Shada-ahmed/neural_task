# Neural Networks Tasks — Perceptron & ADALINE Classification

## Overview

A Python desktop application for binary classification using the **Single-Layer Perceptron (SLP)** and **ADALINE (Adaptive Linear Neuron)** learning algorithms with the Palmer Penguins dataset.

The application allows users to select two features and two classes, configure the learning rate, maximum epochs, and bias, and train and evaluate either algorithm through an interactive graphical user interface.

ADALINE additionally supports an MSE (Mean Squared Error) threshold to control its training stopping condition.

## Features

- Interactive desktop GUI built with PySide6
- Binary classification using Perceptron and ADALINE
- Algorithm selection through the GUI
- Dropdown-based model configuration
- Configurable learning rate, maximum epochs, and bias
- Configurable MSE threshold for ADALINE
- Training and testing accuracy
- Manually implemented confusion matrix
- Accuracy gap analysis
- Decision-boundary visualization using test samples
- Perceptron training-error plot per epoch
- ADALINE MSE plot per epoch
- Automatic training termination based on the algorithm's stopping condition or maximum epochs
- Experiment history
- Export experiment results to CSV
- Data preprocessing and feature standardization

## Algorithms

### 1. Single-Layer Perceptron (SLP)

The Perceptron is a supervised learning algorithm for binary classification.

- Predicts class labels of `-1` and `+1`.
- Updates weights when a sample is misclassified.
- Tracks the number of misclassified training samples per epoch.
- Stops when all training samples are correctly classified or the maximum number of epochs is reached.

### 2. ADALINE (Adaptive Linear Neuron)

ADALINE is a supervised learning algorithm that uses a continuous linear output during training and minimizes the Mean Squared Error (MSE).

- Calculates the continuous output using the weighted sum of the input features and bias.
- Updates weights and bias using the continuous prediction error.
- Calculates MSE after each epoch.
- Supports a configurable MSE threshold for early stopping.
- Converts continuous outputs into class labels of `-1` and `+1` during prediction.
- Displays the MSE history throughout training.

## Dataset and Evaluation

The application uses the Palmer Penguins dataset, `penguins.csv`, for binary classification.

- Select two different numerical features.
- Select two different penguin classes.
- Encode the selected classes as `-1` and `+1`.
- Use 30 training samples and 20 testing samples per selected class.
- Standardize the selected features using training-set statistics.
- Calculate accuracy and the confusion matrix manually without relying on scikit-learn metrics.

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- PySide6

## Project Structure

```text
Neural-task/
├── algorithms/
│   ├── perceptron.py
│   └── adaline.py
├── data_processing/
│   └── pre_processing.py
├── evaluation/
│   └── metrics.py
├── gui/
│   ├── main.py
│   └── main_window.py
├── visualization/
│   └── plots.py
├── penguins.csv
├── requirements.txt
└── README.md
```

*The structure above is illustrative; adjust filenames and locations to match the actual repository.*

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Neural-task
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

From the project root:

```bash
python -m gui.main
```

## Usage

1. Launch the application.
2. Select the algorithm: **Perceptron** or **ADALINE**.
3. Select two different features.
4. Select two different penguin classes.
5. Configure the learning rate, maximum epochs, and bias.
6. If ADALINE is selected, configure the MSE threshold.
7. Click **Train and Evaluate**.
8. Review training accuracy, testing accuracy, accuracy gap, and the confusion matrix.
9. Inspect the decision boundary and the algorithm-specific training plot.
10. Repeat experiments with different configurations.
11. Export experiment results to CSV for further analysis.

## Experiment Tracking

The application maintains an experiment history to help compare different configurations and algorithm performance.

Experiments can be exported to CSV for reporting and analysis, including the selected features, classes, hyperparameters, and evaluation results.

## Current Scope

The application supports two learning algorithms:

- **Perceptron (SLP):** Updates weights based on classification mistakes.
- **ADALINE:** Updates weights using continuous errors and minimizes MSE.

Both algorithms support binary classification, configurable training parameters, manual evaluation metrics, and decision-boundary visualization.
