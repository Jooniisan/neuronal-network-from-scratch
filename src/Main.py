import csv
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

from lab1.src.ModelsNN.knn import KNN
from lab1.src.ModelsNN.multiple_layer_perceptron import MultipleLayerPerceptron
from lab1.src.ModelsNN.svm import SVM
from lab1.src.neural_network import NeuralNetwork

WIDTH = 160
HEIGHT = 120


def create_csv_dataset():
    print('Creating CSV Dataset!')
    path = Path.cwd()
    image_dataset_a_path = path / 'ImageDataset' / 'EnsembleA'
    csv_input_path = path / 'ImageDataset' / 'csv' / 'input-csv'
    csv_output_path = path / 'ImageDataset' / 'csv' / 'output-csv' / 'All_BoundingBox.csv'

    with open(csv_output_path, 'w', newline='') as f:
        fieldnames = ['label', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'path_image']
        the_writer = csv.DictWriter(f, fieldnames=fieldnames)
        the_writer.writeheader()

        for csv_filename in csv_input_path.glob('*'):
            with open(csv_filename) as file:
                reader = csv.reader(file)

                for row in reader:

                    image_label = row[0]

                    if 'Diamand' in image_label:
                        image_label = image_label.replace('Diamand', 'Diamant')

                    if 'hexagone' in image_label:
                        image_label = image_label.replace('hexagone', 'Hexagone')

                    if image_label == 'No_Shape':
                        label_path = image_label.replace('_', '') + 's/'
                    else:
                        label_path = image_label[:-2] + 's/' + image_label.replace('_', '') + '/'

                    the_writer.writerow({
                        'label': image_label,
                        'x1': int(row[1]),
                        'y1': int(row[2]),
                        'x2': int(row[1]) + int(row[3]),
                        'y2': int(row[2]),
                        'x3': int(row[1]),
                        'y3': int(row[2]) + int(row[4]),
                        'x4': int(row[1]) + int(row[3]),
                        'y4': int(row[2]) + int(row[4]),
                        'path_image': image_dataset_a_path / label_path / row[5]
                    })
    print('Done creating CSV Dataset!')


def create_image_dataset(train_value):
    print(f"Creating Image Dataset with a training dataset of {train_value * 100}%")
    csv_all_images_info_path = Path.cwd() / 'ImageDataset' / 'csv' / 'output-csv' / 'All_BoundingBox.csv'
    image_dataset_b_path = Path.cwd() / 'ImageDataset' / 'EnsembleB'
    data_set = []

    with open(csv_all_images_info_path) as file:
        reader = csv.reader(file)
        next(reader)

        for row in reader:
            # Convert image to grayscale, scale to 160 x 120, and only the important region is saved with the label
            if '_2' in row[0] or '_5' in row[0] or 'No_Shape' in row[0]:
                image = cv2.imread(row[9])
                image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cropped = image_grayscale[int(row[2]):int(row[8]), int(row[1]):int(row[7])]
                resized = cv2.resize(cropped, (WIDTH, HEIGHT))
                data_set.append([resized, row[0]])
    random.shuffle(data_set)

    features_train = []
    features_test = []
    outputs_train = []
    outputs_test = []
    classes = []
    count = 0
    for features, label in data_set:
        if count <= (len(data_set) * train_value):
            features_train.append(features)
            outputs_train.append(label)
        else:
            features_test.append(features)
            outputs_test.append(label)
        if label not in classes:
            classes.append(label)
        count += 1

    features_train = np.array(features_train).reshape(-1, WIDTH * HEIGHT)
    features_test = np.array(features_test).reshape(-1, WIDTH * HEIGHT)

    final_features = {'TRAIN': features_train, 'TEST': features_test}
    final_outputs = {'TRAIN': outputs_train, 'TEST': outputs_test, 'CLASSES': classes}

    pickle_out = open(image_dataset_b_path / 'final_features.pickle', "wb")
    pickle.dump(final_features, pickle_out)
    pickle_out.close()

    pickle_out = open(image_dataset_b_path / 'final_outputs.pickle', "wb")
    pickle.dump(final_outputs, pickle_out)
    pickle_out.close()


def use_svm(x_train, x_test, outputs_train, outputs_test):
    my_svm = SVM()
    my_svm.train_svm(x_train, outputs_train)
    prediction = my_svm.predict_svm(x_test)
    my_svm.show_results(prediction, outputs_test)


def use_knn(x_train, x_test, outputs_train, outputs_test):
    my_knn = KNN()
    my_knn.train_knn(x_train, outputs_train)
    prediction = my_knn.predict_knn(x_test)
    my_knn.show_results(prediction, outputs_test)


def use_mlp(x_train, x_test, outputs_train, outputs_test):
    my_mlp = MultipleLayerPerceptron()
    my_mlp.train_mlp(x_train, outputs_train)
    prediction = my_mlp.predict_mlp(x_test)
    my_mlp.show_results(prediction, outputs_test)


def use_my_rn(x_train, x_test, outputs_train, outputs_test, classes):
    my_rn = NeuralNetwork(len(x_train[0]), 960, len(classes), 0.1)
    my_rn.train(x_train, outputs_train, 10)
    prediction = my_rn.predict(x_test)
    my_rn.show_results(prediction, outputs_test)


def main():
    print("Welcome to our Neuronal Network!")

    # Creating CSV which retrieves the correct info useful for creating the dataset
    create_csv_dataset()

    # Create the dataset separating the training and testing data
    training_dataset_value = 0.8

    # while training_dataset_value > 0.2:

    create_image_dataset(training_dataset_value)
    path_pickle = Path.cwd() / 'ImageDataset' / 'EnsembleB'
    with open(path_pickle / 'final_features.pickle', 'rb') as db:
        features = pickle.load(db)

    with open(path_pickle / 'final_outputs.pickle', 'rb') as db:
        outputs = pickle.load(db)

    x_train, x_test = features['TRAIN'], features['TEST']
    outputs_train, outputs_test, classes = outputs['TRAIN'], outputs['TEST'], outputs['CLASSES']

    # Train our neuronal network build from scratch
    # use_my_rn(x_train, x_test, outputs_train, outputs_test, classes)

    # Train a neuronal network using the SVM model
    use_svm(x_train, x_test, outputs_train, outputs_test)

    # Train a neuronal network using the KNN model
    # use_knn(x_train, x_test, outputs_train, outputs_test)

    # Train a neuronal network using the a multiple layer perceptron model
    # use_mlp(x_train, x_test, outputs_train, outputs_test)

    # training_dataset_value -= 0.2


if __name__ == "__main__":
    main()
