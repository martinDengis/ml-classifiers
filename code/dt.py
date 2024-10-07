"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os


# (Question 1): Decision Trees
def train_and_evaluate(X_train, y_train, X_test, y_test, depths):
    """
    Train and evaluate a Decision Tree Classifier with varying depths.

    Parameters
    ----------
    X_train: (array-like)
        Training feature set.
    y_train: (array-like)
        Training labels.
    X_test: (array-like)
        Test feature set.
    y_test: (array-like)
        Test labels.
    depths: (list)
        List of depths to train the Decision Tree Classifier with.

    Returns
    -------
    None

    This function trains a Decision Tree Classifier on the provided training data for each depth in the `depths` list.
    It evaluates the classifier on the test data and prints the accuracy, precision, recall, F1-score, and confusion matrix
    for each depth. It also plots the confusion matrix and decision boundary for each depth.
    """
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train, y_train)  # Fit model on training data

        # Predict labels for test set
        y_pred = clf.predict(X_test)

        # Print evaluation metrics for current depth
        accuracy, precision, recall, f1, cm, cm_normalized = evaluation_metrics(y_test, y_pred)
        f_metrics = (
            f"- Accuracy: {accuracy:.3f};\n"
            f"- Precision: {precision:.3f};\n"
            f"- Recall: {recall:.3f};\n"
            f"- F1-Score: {f1:.3f}"
        )
        print(f"Max Depth: {depth}\n{f_metrics}")

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
        disp.plot()
        plt.title(f"Confusion Matrix for max_depth={depth}")
        # plt.show()
        print(f"Confusion Matrix for max_depth={depth}:\n{cm}")

        # Plot the decision boundary
        output_dir = "out"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_boundary(f"{output_dir}/boundary_depth_{depth}", clf, X_train, y_train, title=f"Decision Boundary for max_depth = {depth}")
        disp.figure_.savefig(f"{output_dir}/confusion_matrix_depth_{depth}.png")


def evaluation_metrics(y_test, y_pred):
    """
    Evaluate the performance of a classification model using various metrics.

    Parameters
    ----------
    y_test: (array-like)
        True labels of the test data.
    y_pred: (array-like)
        Predicted labels by the model.

    Return
    ------
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
    return accuracy, precision, recall, f1, cm, cm_normalized


if __name__ == "__main__":
    # Step 1: Generate the dataset
    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)

    # Print the number of positive and negative examples
    print('Number of positive examples:', np.sum(y_train))
    print('Number of negative examples:', np.sum(y_test == 0))

    # Step 2: Define depths and call the function
    depths = [1, 2, 4, 8, None]
    train_and_evaluate(X_train, y_train, X_test, y_test, depths)

