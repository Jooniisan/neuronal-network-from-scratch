import numpy as np
import math


def step(self, z):
    f = np.where(z >= 0.9, 1, 0)
    return f


def step_diff(self, f):
    f_prime = np.ones(f.shape)
    return f_prime


def sigmoid(self, z):
    sigma = 1 / (1 + (math.exp(z)))
    return sigma


def sigmoid_diff(self, f):
    sigma_prime = self.sigmoid(f) * (1 - self.sigmoid(f))
    return sigma_prime


class Perceptron:
    def __init__(self, learning_rate=0.1, iterations=10, function='step'):
        self.learning_rate = learning_rate
        self.iterations = iterations

        activation = {
            'sigmoid': sigmoid,
            'step': step
        }

        diff = {
            'sigmoid': sigmoid_diff,
            'step': step_diff
        }

        self.activation = activation.get(function, lambda: 'Invalid function')
        self.diff = diff.get(function, lambda: 'Invalid function')

    def forward(self, X):
        z = np.dot(X, self.w) + self.b
        y_pred = self.activation(z)
        return y_pred

    def predict(self, X):
        return self.forward(X)

    def update_weights(self, X, y, y_pred):
        error = y - y_pred
        delta = self.learning_rate * error * self.diff(y_pred)
        self.w += X.T.dot(delta)
        self.b += sum(delta)
        self.train_errors.append(np.count_nonzero(delta))
        self.loss.append(np.mean(np.square(y - y_pred)))

    def train(self, X, y):
        self.w = np.zeros((X.shape[1], 1))
        self.b = np.ones((1, 1))
        self.train_errors = []
        self.loss = []

        for _ in range(self.iterations):
            y_pred = self.forward(X)
            self.update_weights(X, y, y_pred)
        return self

