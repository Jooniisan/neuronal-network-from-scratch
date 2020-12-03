import time

import numpy as np
from tabulate import tabulate
from sklearn import metrics


class NeuralNetwork:
    def __init__(self, nb_input_nodes, nb_hidden_nodes, nb_output_nodes, learning_rate):
        self.nb_input_nodes = nb_input_nodes
        self.nb_hidden_nodes = nb_hidden_nodes
        self.nb_output_nodes = nb_output_nodes
        self.learning_rate = learning_rate
        self.loss = []
        self.W1 = np.random.randn(self.nb_input_nodes, self.nb_hidden_nodes)
        self.b1 = np.ones(self.nb_hidden_nodes)
        self.W2 = np.random.randn(self.nb_hidden_nodes, self.nb_output_nodes)
        self.b2 = np.ones(self.nb_output_nodes)

    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def derivative_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def entropy_loss(self, y, y_pred):
        eps = np.finfo(float).eps
        return -np.sum(y * np.log(y_pred + eps))

    def forward(self, X):
        self.A1 = X
        self.Z2 = np.dot(self.A1, self.W1) + self.b1
        self.A2 = self.sigmoid(self.Z2)
        self.Z3 = np.dot(self.A2, self.W2) + self.b2
        self.A3 = self.sigmoid(self.Z3)
        return self.A3

    def backward(self, x_train, y):
        m = x_train.shape[0]

        dZ3 = self.A3 - y
        dW2 = (1. / m) * np.dot(self.A2.T, dZ3)
        db2 = np.sum(dZ3, axis=0)
        dA2 = np.dot(dZ3, self.W2.T)
        dZ2 = dA2 * self.derivative_sigmoid(self.Z2)
        dW1 = (1. / m) * np.dot(x_train.T, dZ2)
        db1 = (1. / m) * np.sum(dZ2, axis=0)

        self.W2 = self.W2 - self.learning_rate * dW2
        self.b2 = self.b2 - self.learning_rate * db2
        self.W1 = self.W1 - self.learning_rate * dW1
        self.b1 = self.b1 - self.learning_rate * db1

    def train(self, x_train, y_train, nb_iterations):
        start = time.time()
        for i in range(nb_iterations):
            y_pred = self.forward(x_train)
            loss = self.entropy_loss(y_train, y_pred)
            self.loss.append(loss)
            self.backward(x_train, y_train)

            if i == 0 or i == nb_iterations - 1:
                print(f"Iteration: {i + 1}")
                print(tabulate(zip(x_train, y_train, [np.round(y_pred) for y_pred in self.A3]),
                               headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")
                print("\n")
        stop = time.time()
        self.training_time = stop - start

    def predict(self, x_test):
        start = time.time()
        prediction = np.round(self.forward(x_test))
        stop = time.time()
        self.predict_time = stop - start
        return prediction

    def show_results(self, prediction, y_test):
        print('\nNEURONAL NETWORK CLASSIFICATION REPORT:\n%s\n' % (metrics.classification_report(y_test, prediction)))
        print('NEURONAL NETWORK CONFUSION MATRIX: \n%s' % metrics.confusion_matrix(y_test, prediction))
        print(f"TRAINING TIME: {self.training_time}s")
        print(f"PREDICTION TIME: {self.predict_time}s\n")
