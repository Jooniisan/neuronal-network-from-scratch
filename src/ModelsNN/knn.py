import time

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier


class KNN:
    def __init__(self):
        self.knn = KNeighborsClassifier()

    def train_knn(self, x_train, y_train):
        start = time.time()
        self.knn.fit(x_train, y_train)
        stop = time.time()
        self.training_time = stop - start

    def predict_knn(self, x_test):
        start = time.time()
        prediction = self.knn.predict(x_test)
        stop = time.time()
        self.predict_time = stop - start
        return prediction

    def show_results(self, prediction, y_test):
        print('\nKNN CLASSIFICATION REPORT%s:\n%s\n' % (self.knn, metrics.classification_report(y_test, prediction)))
        print('KNN CONFUSION MATRIX: \n%s' % metrics.confusion_matrix(y_test, prediction))
        print(f"TRAINING TIME: {self.training_time}s")
        print(f"PREDICTION TIME: {self.predict_time}s\n")
