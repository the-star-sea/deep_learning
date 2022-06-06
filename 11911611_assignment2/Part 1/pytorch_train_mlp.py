from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from matplotlib.ticker import MultipleLocator
from pytorch_mlp import MLP
import torch
from torch.autograd import Variable
import torch.nn as nn
from modules import CrossEntropy
from sklearn import datasets
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 20
EVAL_FREQ_DEFAULT = 2

FLAGS = None
# Generate dataset

x, t = datasets.make_moons(2000, shuffle=True)
# plt.scatter(x[:,0],x[:,1])
train_x = x[:1400]
train_t = t[:1400]
test_x = x[1400:]
test_t = t[1400:]


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
    for i in range(len(predictions)):
        if predictions[i][0][0] > predictions[i][0][1] and targets[i][0] == 0:
            hit = hit + 1
        elif predictions[i][0][0] < predictions[i][0][1] and targets[i][0] == 1:
            hit = hit + 1
    return (1.0 * hit / len(predictions))


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


def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    n_inputs = len(train_x[0])

    n_hidden = list(map(int, FLAGS.dnn_hidden_units.split()))

    n_classes = 2

    mlp = MLP(n_inputs, n_hidden, n_classes)
    predictions_train = []
    labels_train = []
    train_losses = []
    train_losseses = []
    train_acces = []

    test_losses = []
    steps = []
    test_acces = []
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    for epoch in range(0,FLAGS.max_steps):
        if epoch > 0 and epoch % FLAGS.eval_freq == 0:
            mlp.eval()
            steps.append(epoch)
            accu = accuracy(predictions_train, labels_train)
            l = sum(train_losses) / len(train_losses)
            train_losseses.append(l)
            train_acces.append(accu)
            train_losses = []
            predictions_test = []

            labels_test = []
            eval_losses = []
            for i in range(len(test_x)):
                data_t = torch.Tensor([test_x[i]])
                labels_t = torch.LongTensor([test_t[i]])
                outputs_t = mlp(data_t)

                loss_t = criterion(outputs_t, labels_t)

                eval_losses.append(loss_t.item())

            predictions_test.append(outputs.detach().numpy())
            labels_test.append(labels.detach().numpy())
            eval_acc = accuracy(predictions_test, labels_test)
            eval_loss = sum(eval_losses) / len(eval_losses)
            test_losses.append(eval_loss)
            test_acces.append(eval_acc)
        mlp.train()
        train_loss = 0
        for i in range(len(train_x)):
            data = torch.Tensor([train_x[i]])
            labels = torch.LongTensor([train_t[i]])
            optimizer.zero_grad()
            outputs = mlp(data)

            loss = criterion(outputs, labels)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        train_losses.append(train_loss / len(train_x))
        predictions_train.append(outputs.detach().numpy())
        labels_train.append(labels.detach().numpy())
    graph(steps, train_acces, test_acces, train_losseses, test_losses)
    return


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
