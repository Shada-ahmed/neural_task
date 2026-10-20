
import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=100, use_bias=True , random_state=42):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.use_bias = use_bias
        self.rng = np.random.default_rng(random_state)
        # Exactly two weights because we use two features
        self.weights = np.random.uniform(-0.01, 0.01, 2)
        self.bias = np.random.uniform(-0.01, 0.01) if use_bias else 0

        self.errors_per_epoch = []

    ''' def predict(self, x):
        # x contains exactly two feature values
        result = np.dot(x, self.weights) + self.bias

        if result >= 0:
            return 1
        return -1'''

    def predict(self, X):
        X = np.asarray(X)
        results = np.dot(X, self.weights) + self.bias
        return np.where(results >= 0, 1, -1)

    def fit(self, X, y):
        
        self.errors_per_epoch = []

        for epoch in range(self.epochs):
            errors = 0

# Here we loop over each sample per epoch
            # Shuffle the sample indices
            indices = self.rng.permutation(len(X))
            for i in indices:
                prediction = self.predict(X[i])

                if prediction != y[i]:
                    update = self.learning_rate * (y[i] - prediction)

                    self.weights = self.weights + update * X[i]

                    if self.use_bias:
                        self.bias = self.bias + update

                    errors += 1

 # Store the number of errors per epoch to visualize the learning process
            self.errors_per_epoch.append(errors)

            # Stop if every sample is classified correctly
            if errors == 0:
                break

        return self
