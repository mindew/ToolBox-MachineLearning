""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression


def display_digits():
    digits = load_digits()
    print(digits.DESCR)
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    num_trials = 10
    train_percentages = range(5, 95, 5)
    # returns the numbers between 5 and 95 with the step size of 5
    test_accuracies = numpy.zeros(len(train_percentages))
    # train models with training percentages between 5 and 90 (see
    # train_percentages) and evaluate the resultant accuracy for each.
    # You should repeat each training percentage num_trials times to smooth out
    # variability.
    # For consistency with the previous example use
    # model = LogisticRegression(C=10**-10) for your learner
    for i, size in enumerate(train_percentages):
        A = numpy.zeros(10)
        # returns an array of zeros
        for k in range(10):
            X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size = size/1000)

            model = LogisticRegression(C=10**-2)
            # with LogisticRegression (C = 10**-10)
            # the curve was very spikey
            model.fit(X_train, y_train)
            A[k] = model.score(X_test, y_test)
            # takes feature matrix X_test and the expected target
            # values y_test. Predictions for X_test are compared
            # with y_test and either accuracy (for classifiers) or R^2
            # score (for regression estimators is returned)
        accuracies = numpy.mean(A)
        # computes the average of the array elements
        test_accuracies[i] = accuracies
        print("Test Accuracy %f" % accuracies)

    # plotting the curve
    fig = plt.figure()
    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    plt.show()


if __name__ == "__main__":
    # Feel free to comment/uncomment as needed
    # display_digits()
    train_model()
