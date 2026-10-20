
# Neural Networks Lab 3 — Perceptron Classification

## Overview

A Python desktop application for binary classification using the
Perceptron learning algorithm and the Palmer Penguins dataset.

The application allows users to select two features and two classes,
configure the learning rate, maximum epochs, and bias, and evaluate
the trained model.

## Features

- Interactive GUI built with PySide6
- Binary classification using Perceptron
- Dropdown-based model configuration
- Training and testing accuracy
- Manually implemented confusion matrix
- Decision-boundary visualization using test samples
- Training errors per epoch
- Experiment history
- Export experiment results to CSV

## Technologies

- Python
- NumPy
- Pandas
- Matplotlib
- PySide6

## Project Structure

- `algorithms/` — Perceptron implementation
- `data_processing/` — Data loading and preprocessing
- `evaluation/` — Accuracy and confusion matrix calculations
- `gui/` — Desktop application
- `visualization/` — Plotting utilities
- `penguins.csv` — Dataset

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd Neural-task
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

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

1. Select two different features.
2. Select two different penguin classes.
3. Choose the learning rate, epochs, and bias setting.
4. Click **Train and Evaluate**.
5. Review the performance metrics, confusion matrix, and plots.
6. Export experiment results to CSV.

## Current Scope

The current implementation focuses on the Perceptron algorithm.
Adaline is not implemented yet.
