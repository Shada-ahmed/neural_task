
import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, weights, bias, feature1, feature2, class1, class2):
  
    """
    Plot two classes and their learned linear decision boundary.

    X must contain exactly two features.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    weights = np.asarray(weights)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must contain exactly two features.")

    if len(weights) != 2:
        raise ValueError("Exactly two weights are required.")

    # Separate the samples by their class labels.
    class1_points = X[y == -1]
    class2_points = X[y == 1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
     class1_points[:, 0],
        class1_points[:, 1],
        label=class1,
        marker="o",
    )

    plt.scatter(
        class2_points[:, 0],
        class2_points[:, 1],
        label=class2,
        marker="x",
    )

    # Decision boundary:
    # w1*x1 + w2*x2 + bias = 0
    x_values = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

    if abs(weights[1]) > 1e-12:
        y_values = (
            -weights[0] * x_values - bias
        ) / weights[1]

        plt.plot(
            x_values,
            y_values,
            label="Decision boundary",
        )

    elif abs(weights[0]) > 1e-12:
        # Vertical boundary when the second weight is zero.
        x_boundary = -bias / weights[0]

        plt.axvline(
            x=x_boundary,
            label="Decision boundary",
        )

    else:
        print("Cannot draw a decision boundary: both weights are zero.")

    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.title(f"{class1} vs {class2}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_errors(errors_per_epoch):
    """Plot the number of training errors in each epoch."""
    epochs = range(1, len(errors_per_epoch) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, errors_per_epoch, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Number of errors")
    plt.title("Perceptron Training Errors")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
