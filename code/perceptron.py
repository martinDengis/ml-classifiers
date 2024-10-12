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
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os


# (Question 3): Perceptron


class PerceptronClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_iter=5, learning_rate=.0001):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        """Fit a perceptron model on (X, y)

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """

        # Input validation
        X = np.asarray(X, dtype=np.float64)
        n_instances, n_features = X.shape

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("This class is only dealing with binary "
                             "classification problems")

        # TODO: fill in this function
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(n_instances), X])
        
        # Initialize weights randomly with small values
        self.weights = np.random.randn(n_features + 1) * 0.01
        
        # Stochastic gradient descent
        for _ in range(self.n_iter):
            # Shuffle the data for each epoch
            indices = np.random.permutation(n_instances)
            X_shuffled = X_with_bias[indices]
            y_shuffled = y[indices]
            
            # Update weights for each training example
            for xi, yi in zip(X_shuffled, y_shuffled):
                # Forward pass
                z = np.dot(xi, self.weights)
                prediction = self.sigmoid(z)
                
                # Compute gradient and update weights
                error = prediction - yi
                gradient = error * xi
                self.weights -= self.learning_rate * gradient

        return self


    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """

        # TODO: fill in this function
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """

        # TODO: fill in this function
        if self.weights is None:
            raise ValueError("Model has not been fitted yet.")
            
        # Add bias term to X
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Compute probabilities
        z = np.dot(X_with_bias, self.weights)
        proba_class_1 = self.sigmoid(z)
        
        # Return probabilities for both classes
        return np.column_stack([1 - proba_class_1, proba_class_1])

if __name__ == "__main__":    
    # Create output directory if it doesn't exist
    os.makedirs("out/Q3", exist_ok=True)
    
    # Learning rates to test
    learning_rates = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]
    
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate dataset with fixed random state for reproducibility
    random_state_train = 42
    random_state_test = 43
    X_train, y_train = make_dataset(1000, random_state=random_state_train)
    X_test, y_test = make_dataset(2000, random_state=random_state_test)
    # 3.a) Generate decision boundary plots for different learning rates
    for eta in learning_rates:
        # Train classifier
        clf = PerceptronClassifier(n_iter=5, learning_rate=eta)
        clf.fit(X_train, y_train)
        
        # Plot and save decision boundary
        filename = os.path.join("out/Q3", f"perceptron_boundary_eta_{eta:.0e}")
        plot_boundary(filename, clf, X_test, y_test, 
                     title=f"Decision Boundary (η={eta:.0e})")
        
        y_pred = clf.predict(X_test)

        # Compute, plot and save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix (η={eta:.0e})')
        plt.savefig(os.path.join("out/Q3", f"confusion_matrix_eta_{eta:.0e}.pdf"))
        plt.close()
    
    # 4) Compute average accuracies and standard deviations
    n_runs = 5
    accuracies = {eta: [] for eta in learning_rates}
    
    for run in range(n_runs):
        # Generate new datasets for each run with different but reproducible random states
        X_train, y_train = make_dataset(1000, random_state=42 + run)
        X_test, y_test = make_dataset(2000, random_state=43 + run)
        
        for eta in learning_rates:
            # Train classifier
            clf = PerceptronClassifier(n_iter=5, learning_rate=eta)
            clf.fit(X_train, y_train)
            
            # Compute accuracy
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies[eta].append(acc)

    # Print results
    print("\nPerceptron Classification Results:")
    print("---------------------------------")
    print("Learning Rate | Mean Accuracy ± Std")
    print("---------------------------------")
    for eta in learning_rates:
        mean_acc = np.mean(accuracies[eta])
        std_acc = np.std(accuracies[eta])
        print(f"{eta:11.0e} | {mean_acc:.3f} ± {std_acc:.3f}")
