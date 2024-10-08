"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt

from data import make_dataset
from plot import plot_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score


# (Question 1): Decision Trees
output_dir = "out/q1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def train_and_evaluate(X_train, y_train, X_test, y_test, depths, eval_metrics=True):
    """
    Train and evaluate a Decision Tree Classifier with varying depths.

    Args:
        X_train (array-like): Training data.
        y_train (array-like): Training labels.
        X_test (array-like): Test data.
        y_test (array-like): Test labels.
        depths (list): A list of integers representing the depths of the decision trees to be evaluated.
        eval_metrics (bool): Whether to evaluate the model using various metrics.

    Returns:
        dict: A dictionary where keys are tree depths and values are the accuracies of the classifier.
    """
    accuracies = {depth: 0 for depth in depths} # Initialize dictionary to store accuracies
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)   # Initialize Decision Tree Classifier
        clf.fit(X_train, y_train)  # Fit model on training data
        y_pred = clf.predict(X_test)    # Predict labels for test set
        plot_boundary(f"{output_dir}/boundary_depth_{depth}", clf, X_test, y_test, title=f"Decision Boundary for max_depth = {depth}")

        # Evaluation metrics for current depth
        accuracy = accuracy_score(y_test, y_pred)   # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracies[depth] = accuracy

        if eval_metrics:
            evaluation_metrics(y_test, y_pred, depth)

    return accuracies


def accuracies_reporting(depths, n_generation=5):
    """
    Computes and reports the accuracies of a classifier for different tree depths over multiple generations.

    Args:
        depths (list): A list of integers representing the depths of the decision trees to be evaluated.
        n_generation (int): The number of generations to run the experiment.

    Returns:
        dict: A dictionary where keys are tree depths and values are lists of accuracies for each generation.
    """
    accuracies = {depth: [] for depth in depths}
    for _ in range(n_generation):
        X_train, y_train = make_dataset(n_points=1000)
        X_test, y_test = make_dataset(n_points=2000)
        gen_accuracies = train_and_evaluate(X_train, y_train, X_test, y_test, depths, eval_metrics=False)

        for depth in depths:
            accuracies[depth].append(gen_accuracies[depth])

    return accuracies


def evaluation_metrics(y_test, y_pred, depth):
    """
    Evaluate the performance of a classification model using various metrics.

    Args:
        y_test (array-like): True labels of the test data.
        y_pred (array-like): Predicted labels by the model.
        depth (int): The depth of the decision tree.

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
    print(f"Max Depth: {depth}\n{f_metrics}")

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
    disp.plot()
    plt.title(f"Confusion Matrix for max_depth={depth}")
    disp.figure_.savefig(f"{output_dir}/confusion_matrix_depth_{depth}.png")
    plt.close()

    return accuracy, precision, recall, f1, cm, cm_normalized


if __name__ == "__main__":
    depths = [1, 2, 4, 8, None]

    # Q1.1
    print("Q1.1\n----------")
    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    train_and_evaluate(X_train, y_train, X_test, y_test, depths, eval_metrics=True)

    # Q1.2
    print("Q1.2\n----------")
    accuracies = accuracies_reporting(depths, 5)
    for depth in depths:
        avg_accuracy = np.mean(accuracies[depth])
        std_accuracy = np.std(accuracies[depth])
        print(f"Max Depth: {depth}")
        print(f"Average Accuracy: {avg_accuracy:.3f}")
        print(f"Standard Deviation: {std_accuracy:.3f}\n")

