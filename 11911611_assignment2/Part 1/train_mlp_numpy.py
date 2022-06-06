from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
from matplotlib import pyplot as plt
import numpy as np
import os

from matplotlib.ticker import MultipleLocator

from mlp_numpy import MLP
from modules import CrossEntropy
from sklearn import datasets
# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 100
EVAL_FREQ_DEFAULT = 10

FLAGS = None
GD="SGD"

def onehot(label):
    onehot = np.zeros((len(label), 2))
    for i in range(len(label)):
        onehot[i][0] = label[i]
        onehot[i][1] = 1-label[i]
    return onehot
steps, label = datasets.make_moons(n_samples=2000, shuffle=True)
# plt.scatter(x[:,0], x[:,1])
# plt.show()
label =onehot(label)
train_x = steps[:1400]
train_label = label[:1400]
test_x = steps[1400:]
test_label = label[1400:]
def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    hit = 0
    length=len(predictions)
    for i in range(length):
        if (np.around(predictions[i]) == targets[i]).all():
            hit += 1
    accuracy = hit / length
    return accuracy
def loss_accuracy(mlp, data, label):
    loss = 0
    predictions = []
    for i in range(len(label)):
        label_t,data_t = resize(label[i],data[i])

        out = mlp.forward(data_t)
        loss += CrossEntropy().forward(out, label_t)
        predictions.append(out)
    return accuracy(predictions, label), loss/len(data)

def resize(X,y):
    return X.reshape(1, -1),y.reshape(1, -1)
def shuffle(X, y):
    randomlist = np.arange(len(y))
    np.random.shuffle(randomlist)
    return X[randomlist], y[randomlist]
def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    mlp = MLP(len(train_x[0]), list(map(int, FLAGS.dnn_hidden_units.split())), len(train_label[0]))
    accu_train = []
    accu_test = []
    loss_train = []
    loss_test = []
    steps = []
    for step in range(FLAGS.max_steps):
        shuffle(train_x, train_label)
        for i in range(len(train_x)):
            data,label=resize(train_x[i],train_label[i])
            dout = mlp.forward(data) - label
            mlp.backward(dout,FLAGS.learning_rate, FLAGS.grad_type)
        if FLAGS.grad_type == 'BGD':
            mlp.update(FLAGS.learning_rate, len(train_x))

        if step % FLAGS.eval_freq == 0:
            steps.append(step)
            accu, loss = loss_accuracy(mlp, train_x, train_label)
            accu_train.append(accu)
            loss_train.append(loss)
            accu, loss = loss_accuracy(mlp, test_x, test_label)
            accu_test.append(accu)
            loss_test.append(loss)
    graph(steps, accu_train, accu_test, loss_train, loss_test)
    return
def graph(steps, accu_train, accu_test, loss_train, loss_test):
    fig1 = plt.subplot(2, 1, 1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0.3, 1.2)
    fig2 = plt.subplot(2, 1, 2)
    fig1.plot(steps, accu_train, c='red', label='training data accuracy')
    fig1.plot(steps, accu_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(steps, loss_train, c='green', label='train  loss')
    fig2.plot(steps, loss_test, c='yellow', label='test  loss')
    fig2.legend()
    plt.show()
def main():
    """
    Main function
    """

    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    parser.add_argument('--grad_type', type=str, default=GD,
                        help=' Batch gradient descent or stochastic gradient descent')
    FLAGS, unparsed = parser.parse_known_args()
    main()