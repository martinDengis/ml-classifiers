"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from data import make_dataset
import numpy as np
from perceptron import PerceptronClassifier

# (Question 4): Method comparison


# Constants
RANDOM_SEED = 42
k = 10  # Number of folds for cross-validation
n_generations = 5  # Number of dataset generations

# Hyperparameters
depths = [1, 2, 4, 8, None]
n_neighbors_list = [1, 5, 50, 100, 500]
learning_rates = [1e-4, 5e-4, 1e-3, 1e-2, 1e-1]

def evaluate_model(clf, X_train, y_train, kf):
    """
    Evaluate a model using K-Fold Cross-Validation.

    Args:
        clf (object): Classifier instance from scikit-learn.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.
        kf (KFold): KFold cross-validator instance.

    Returns:
        float: The mean accuracy over the k folds.
    """
    accuracies = []
    for train_index, val_index in kf.split(X_train):
        X_fold_train, X_fold_val = X_train[train_index], X_train[val_index]
        y_fold_train, y_fold_val = y_train[train_index], y_train[val_index]
        clf.fit(X_fold_train, y_fold_train) # Fit the model on the training fold
        y_pred = clf.predict(X_fold_val) # Predict on the validation fold
        accuracy = accuracy_score(y_fold_val, y_pred) # Calculate accuracy
        accuracies.append(accuracy) # Store the accuracy
    return np.mean(accuracies)

def find_best_hyperparameters():
    """
    Find the best hyperparameters for each model using only the learning set and cross-validation.

    Returns:
        dict: Dictionary containing the best hyperparameters for each model.
        dict: Dictionary containing all hyperparameter values and their performances.
    """
    best_hyperparams = {}
    all_hyperparams_results = {}
    X_train, y_train = make_dataset(n_points=1000, random_state=RANDOM_SEED)

    # K-Fold Cross-Validation
    kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)

    # Decision Tree
    best_depth = None
    best_accuracy = 0
    results = {}
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_SEED)
        accuracy = evaluate_model(clf, X_train, y_train, kf)
        results[depth] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
    best_hyperparams['DecisionTree'] = best_depth
    all_hyperparams_results['DecisionTree'] = results

    # KNN
    best_n_neighbors = None
    best_accuracy = 0
    results = {}
    for n_neighbors in n_neighbors_list:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        accuracy = evaluate_model(clf, X_train, y_train, kf)
        results[n_neighbors] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_n_neighbors = n_neighbors
    best_hyperparams['KNN'] = best_n_neighbors
    all_hyperparams_results['KNN'] = results

    # Perceptron
    best_eta = None
    best_accuracy = 0
    results = {}
    for eta in learning_rates:
        clf = PerceptronClassifier(n_iter=5, learning_rate=eta)
        accuracy = evaluate_model(clf, X_train, y_train, kf)
        results[eta] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_eta = eta
    best_hyperparams['Perceptron'] = best_eta
    all_hyperparams_results['Perceptron'] = results

    return best_hyperparams, all_hyperparams_results

def test_model_accuracy(best_hyperparams, n_irr=0):
    """
    Test the model using the best hyperparameters on the test set and calculate the average accuracy over multiple generations.

    Args:
        best_hyperparams (dict): Best hyperparameters found for each model.
        n_irr (int): Number of irrelevant (noise) features to add to the dataset.

    Returns:
        dict: Dictionary containing average test set accuracy and standard deviation for each model.
    """
    results = {
        'DecisionTree': [],
        'KNN': [],
        'Perceptron': []
    }

    for generation in range(n_generations):
        # Generate dataset
        X_train, y_train = make_dataset(n_points=1000, random_state=RANDOM_SEED + generation, n_irrelevant=n_irr)
        X_test, y_test = make_dataset(n_points=2000, random_state=RANDOM_SEED + generation, n_irrelevant=n_irr)

        # Decision Tree
        clf = DecisionTreeClassifier(max_depth=best_hyperparams['DecisionTree'], random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['DecisionTree'].append(accuracy)

        # KNN
        clf = KNeighborsClassifier(n_neighbors=best_hyperparams['KNN'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['KNN'].append(accuracy)

        # Perceptron
        clf = PerceptronClassifier(n_iter=5, learning_rate=best_hyperparams['Perceptron'])
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results['Perceptron'].append(accuracy)

    # Calculate mean and standard deviation for each model
    summary = {}
    for model, accuracies in results.items():
        summary[model] = {
            'mean_accuracy': np.mean(accuracies),
            'std_deviation': np.std(accuracies)
        }

    return summary

if __name__ == "__main__":
    # Q4.2 
    print("Q4.2\n----------")
    # Find the best hyperparameters using the learning set
    best_hyperparams, all_hyperparams_results = find_best_hyperparameters()
    print("\nAll hyperparameter performances based on learning set:")
    for model, results in all_hyperparams_results.items():
        print(f"\n{model} results:")
        for hyperparam_value, accuracy in results.items():
            hyperparam_display = f"{hyperparam_value:.0e}" if isinstance(hyperparam_value, float) else hyperparam_value
            print(f"Hyperparameter: {hyperparam_display}, Accuracy: {accuracy:.3f}")

    # Print the best hyperparameters
    print("\nBest hyperparameters based on learning set:")
    for model, param in best_hyperparams.items():
        hyperparam_display = f"{param:.0e}" if isinstance(param, float) else param
        print(f"{model}: {hyperparam_display}")

    # Test the models on the test set with the best hyperparameters
    results_original = test_model_accuracy(best_hyperparams, n_irr=0)
    print("\nTest set results with only original features:")
    for model, result in results_original.items():
        print(f"{model} - Mean accuracy: {result['mean_accuracy']:.3f}, Std deviation: {result['std_deviation']:.3f}")

    # Q4.3
    print("Q4.3\n----------")
    results_with_noise = test_model_accuracy(best_hyperparams, n_irr=200)
    print("\nTest set results with 200 irrelevant features:")
    for model, result in results_with_noise.items():
        print(f"{model} - Mean accuracy: {result['mean_accuracy']:.3f}, Std deviation: {result['std_deviation']:.3f}")
