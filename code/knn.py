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
from sklearn.metrics import confusion_matrix, accuracy_score
import os


# (Question 2): KNN
output_dir = "out/q2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)



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

def train_and_evaluate(X_train, y_train, X_test, y_test, n_neighbors):
    accuracies = {n: 0 for n in n_neighbors}
    for n in n_neighbors:
        neigh = KNeighborsClassifier(n_neighbors=n)
        neigh.fit(X_train, y_train)
        y_pred = neigh.predict(X_test)

        accuracies[n] = accuracy_score(y_test, y_pred)

        plot_boundary(f"{output_dir}/knn_{n}", neigh, X_test, y_test, title="KNN (k={n})")

    return accuracies


if __name__ == "__main__":
    n_neighbors = [1, 5, 50, 100, 500]

    # Q2.1
    print("Q2.1\n----------")
    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    train_and_evaluate(X_train, y_train, X_test, y_test, n_neighbors)





