import numpy as np


class NeuralNetwork:

    def __init__(
        self,
        input_size=5,
        hidden_size=3,
        output_size=5,
        learning_rate=0.01
    ):

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.reset()

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def reset(self):

        rng = np.random.default_rng()

        # Simplified Xavier/He initialization
        self.W1 = (
            rng.normal(
                0,
                np.sqrt(2 / self.input_size),
                (self.input_size, self.hidden_size)
            )
        )

        self.W2 = (
            rng.normal(
                0,
                np.sqrt(2 / self.hidden_size),
                (self.hidden_size, self.hidden_size)
            )
        )

        self.W3 = (
            rng.normal(
                0,
                np.sqrt(2 / self.hidden_size),
                (self.hidden_size, self.output_size)
            )
        )

        self.b1 = np.zeros(self.hidden_size)
        self.b2 = np.zeros(self.hidden_size)
        self.b3 = np.zeros(self.output_size)

        self.epoch = 0
        self.loss = 0.0

    # ========================================================
    # RELU
    # ========================================================

    def relu(self, x):

        return np.maximum(0, x)

    def relu_derivative(self, x):

        return (x > 0).astype(float)

    # ========================================================
    # SOFTMAX
    # ========================================================

    def softmax(self, x):

        # Subtract the maximum to avoid numerical issues
        # with exp()

        exp_values = np.exp(
            x - np.max(x, axis=1, keepdims=True)
        )

        return (
            exp_values /
            np.sum(exp_values, axis=1, keepdims=True)
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, X):

        # Layer 1
        self.z1 = X @ self.W1 + self.b1

        self.a1 = self.relu(self.z1)

        # Layer 2
        self.z2 = self.a1 @ self.W2 + self.b2

        self.a2 = self.relu(self.z2)

        # Layer 3
        self.z3 = self.a2 @ self.W3 + self.b3

        # z3 contains the LOGITS
        self.logits = self.z3

        # Convert to probabilities
        self.probabilities = self.softmax(self.logits)

        return self.probabilities

    # ========================================================
    # LOSS
    # ========================================================

    def calculate_loss(self, y):

        number_of_samples = len(y)

        correct_probabilities = (
            self.probabilities[
                np.arange(number_of_samples),
                y
            ]
        )

        # Small value to avoid log(0)
        correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0
        )

        loss = -np.mean(
            np.log(correct_probabilities)
        )

        return loss

    # ========================================================
    # BACKPROPAGATION
    # ========================================================

    def backward(self, X, y):

        number_of_samples = len(y)

        # ----------------------------------------------------
        # Logit gradient
        # ----------------------------------------------------

        dz3 = self.probabilities.copy()

        dz3[
            np.arange(number_of_samples),
            y
        ] -= 1

        dz3 /= number_of_samples

        # ----------------------------------------------------
        # W3 and b3
        # ----------------------------------------------------

        dW3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0)

        # ----------------------------------------------------
        # Back to Layer 2
        # ----------------------------------------------------

        da2 = dz3 @ self.W3.T

        dz2 = (
            da2 *
            self.relu_derivative(self.z2)
        )

        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # ----------------------------------------------------
        # Back to Layer 1
        # ----------------------------------------------------

        da1 = dz2 @ self.W2.T

        dz1 = (
            da1 *
            self.relu_derivative(self.z1)
        )

        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # ----------------------------------------------------
        # Update parameters
        # ----------------------------------------------------

        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    # ========================================================
    # TRAINING
    # ========================================================

    def train_one_epoch(self, X, y):

        # Forward
        self.forward(X)

        # Loss
        self.loss = self.calculate_loss(y)

        # Backpropagation
        self.backward(X, y)

        self.epoch += 1

        return self.loss

    # ========================================================
    # PREDICTION
    # ========================================================

    def predict(self, X):

        # Accept a single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = self.forward(X)

        return probabilities[0]

    # ========================================================
    # ACCURACY
    # ========================================================

    def accuracy(self, X, y):

        probabilities = self.forward(X)

        predictions = np.argmax(
            probabilities,
            axis=1
        )

        correct = np.sum(
            predictions == y
        )

        return correct, len(y)
