"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import os


# (Question 2): KNN
output_dir = "out/q2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


def train_and_evaluate(X_train, y_train, X_test, y_test, n_neighbors, eval_metrics=True):
    """
    Train and evaluate a KNN classifier for different values of n_neighbors.

    Args:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        n_neighbors (list): A list of integers representing the number of neighbors to consider.
        eval_metrics (bool): Whether to evaluate the model using various metrics.

    Returns:
        dict: A dictionary where keys are the number of neighbors and values are the accuracies of the model.
    """
    accuracies = {n: 0 for n in n_neighbors}    # Initialize dictionary to store accuracies
    for n in n_neighbors:
        neigh = KNeighborsClassifier(n_neighbors=n)   # Initialize KNN Classifier
        neigh.fit(X_train, y_train) # Fit model on training data
        y_pred = neigh.predict(X_test)  # Predict labels for test set
        plot_boundary(f"{output_dir}/knn_{n}", neigh, X_test, y_test, title=f"KNN (k={n})")

        # Evaluation metrics for current n
        accuracies[n] = accuracy_score(y_test, y_pred)

        if eval_metrics:
            evaluation_metrics(y_test, y_pred, n)

    return accuracies


def accuracies_reporting(n_neighbors, n_generation=5):
    """
    Computes and reports the accuracies of a KNN classifier for different values of n_neighbors over multiple generations.

    Args:
        n_neighbors (list): A list of integers representing the number of neighbors to consider.
        n_generation (int): The number of generations to run the experiment.

    Returns:
        dict: A dictionary where keys are the number of neighbors and values are lists of accuracies over the generations.
    """
    accuracies = {n: [] for n in n_neighbors}
    for _ in range(n_generation):
        X_train, y_train = make_dataset(n_points=1000)
        X_test, y_test = make_dataset(n_points=2000)
        gen_accuracies = train_and_evaluate(X_train, y_train, X_test, y_test, n_neighbors, eval_metrics=False)

        for n in n_neighbors:
            accuracies[n].append(gen_accuracies[n])

    return accuracies


def evaluation_metrics(y_test, y_pred, n):
    """
    Evaluate the performance of a classification model using various metrics.

    Args:
        y_test (array-like): True labels of the test data.
        y_pred (array-like): Predicted labels by the model.
        n (int): The number of neighbors used in the KNN model.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
        - accuracy (float): The accuracy of the model.
        - precision (float): The precision of the model.
        - recall (float): The recall of the model.
        - f1 (float): The F1 score of the model.
        - cm (ndarray): The confusion matrix.
        - cm_normalized (ndarray): The normalized confusion matrix.
    """
    accuracy = accuracy_score(y_test, y_pred)   # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = precision_score(y_test, y_pred) # Precision = TP / (TP + FP)
    recall = recall_score(y_test, y_pred)    # Recall = TP / (TP + FN)
    f1 = f1_score(y_test, y_pred)   # F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Print evaluation metrics
    f_metrics = (
        f"- Accuracy: {accuracy:.3f};\n"
        f"- Precision: {precision:.3f};\n"
        f"- Recall: {recall:.3f};\n"
        f"- F1-Score: {f1:.3f}\n"
    )
    print(f"Number of neighbours: {n}\n{f_metrics}")

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot()
    plt.title(f"Confusion Matrix for n={n}")
    disp.figure_.savefig(f"{output_dir}/confusion_matrix_n_{n}.png")
    plt.close()

    return accuracy, precision, recall, f1, cm, cm_normalized


if __name__ == "__main__":
    n_neighbors = [1, 5, 50, 100, 500]

    # Q2.1
    print("Q2.1\n----------")
    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    train_and_evaluate(X_train, y_train, X_test, y_test, n_neighbors)

    # Q2.2
    print("\nQ2.2\n----------")
    accuracies = accuracies_reporting(n_neighbors)
    



