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


# (Question 2): KNN

# Put your funtions here
# ...


if __name__ == "__main__":
    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train, y_train)
    plot_boundary("out/q2/knn_1", neigh, X_test, y_test, title="KNN (k=1)")

    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X_train, y_train)
    plot_boundary("out/q2/knn_5", neigh, X_test, y_test, title="KNN (k=5)")

    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    neigh = KNeighborsClassifier(n_neighbors=50)
    neigh.fit(X_train, y_train)
    plot_boundary("out/q2/knn_50", neigh, X_test, y_test, title="KNN (k=50)")

    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    neigh = KNeighborsClassifier(n_neighbors=100)
    neigh.fit(X_train, y_train)
    plot_boundary("out/q2/knn_100", neigh, X_test, y_test, title="KNN (k=100)")

    X_train, y_train = make_dataset(n_points=1000)
    X_test, y_test = make_dataset(n_points=2000)
    neigh = KNeighborsClassifier(n_neighbors=500)
    neigh.fit(X_train, y_train)
    plot_boundary("out/q2/knn_500", neigh, X_test, y_test, title="KNN (k=500)")


