import time

from sklearn import svm, metrics


class SVM:
    def __init__(self):
        self.clf = svm.SVC()

    def train_svm(self, x_train, y_train):
        start = time.time()
        self.clf.fit(x_train, y_train)
        stop = time.time()
        self.training_time = stop - start

    def predict_svm(self, x_test):
        start = time.time()
        prediction = self.clf.predict(x_test)
        stop = time.time()
        self.predict_time = stop - start
        return prediction

    def show_results(self, prediction, y_test):
        print('\nSVM CLASSIFICATION REPORT%s:\n%s\n' % (self.clf, metrics.classification_report(y_test, prediction)))
        print('SVM CONFUSION MATRIX: \n%s' % metrics.confusion_matrix(y_test, prediction))
        print(f"TRAINING TIME: {self.training_time}s")
        print(f"PREDICTION TIME: {self.predict_time}s\n")
