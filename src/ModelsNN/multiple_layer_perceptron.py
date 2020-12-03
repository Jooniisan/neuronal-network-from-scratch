import time
import warnings

from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier


class MultipleLayerPerceptron:
    def __init__(self):
        self.mlp = MLPClassifier(hidden_layer_sizes=(150, 100, 50), max_iter=300, activation='relu', solver='adam',
                                 random_state=1)

    def train_mlp(self, x_train, y_train):
        start = time.time()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")
            self.mlp.fit(x_train, y_train)
        stop = time.time()
        self.training_time = stop - start

    def predict_mlp(self, x_test):
        start = time.time()
        prediction = self.mlp.predict(x_test)
        stop = time.time()
        self.predict_time = stop - start
        return prediction

    def show_results(self, prediction, y_test):
        print('MLP CLASSIFICATION REPORT%s:\n%s\n' % (self.mlp, metrics.classification_report(y_test, prediction)))
        print('MLP CONFUSION MATRIX: \n%s' % metrics.confusion_matrix(y_test, prediction))
        print(f"TRAINING TIME: {self.training_time}s")
        print(f"PREDICTION TIME: {self.predict_time}s\n")
