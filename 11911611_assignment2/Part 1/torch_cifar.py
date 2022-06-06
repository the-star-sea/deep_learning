from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.ticker import MultipleLocator
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 10
EVAL_FREQ_DEFAULT = 1

FLAGS = None

train_accuracy = []
test_accuracy = []
train_loss = []
test_loss = []
plot_x = []

class MLP(nn.Module):
    def __init__(self, node, keep_rate=0):
        super(MLP, self).__init__()
        self.n_hidden_nodes = node
        self.n_hidden_layers = 2
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        self.fc1 = torch.nn.Linear(32 * 32 * 3, node)
        self.drop1 = torch.nn.Dropout(1 - keep_rate)
        self.fc2 = torch.nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes)
        self.drop2 = torch.nn.Dropout(1 - keep_rate)
        self.out = torch.nn.Linear(self.n_hidden_nodes, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        if self.n_hidden_layers == 2:
            x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return F.log_softmax(self.out(x), dim=0)

def graph(steps, accu_train, accu_test, loss_train, loss_test):
    fig1 = plt.subplot(2, 1, 1)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, 1)
    fig2 = plt.subplot(2, 1, 2)
    fig1.plot(steps, accu_train, c='red', label='training data accuracy')
    fig1.plot(steps, accu_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(steps, loss_train, c='green', label='train  loss')
    fig2.plot(steps, loss_test, c='yellow', label='test  loss')
    fig2.legend()
    plt.savefig("./cifar.png")
    plt.show()
def main():
    cuda = torch.cuda.is_available()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0, pin_memory=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0, pin_memory=False)

    model = MLP(100)
    if cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    steps=[]
    for epoch in range(1, FLAGS.max_steps + 1):
        model.train()
        correct = 0
        count = 0
        loss_train = 0

        for i_batch, (input, target) in enumerate(train_loader):
            count += 1
            if cuda:
                input, target = input.cuda(), target.cuda()
            input, target = Variable(input), Variable(target)
            optimizer.zero_grad()
            output = model(input)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum()
            loss = F.nll_loss(output, target)
            loss_train += loss
            loss.backward()
            optimizer.step()
        if epoch % FLAGS.eval_freq == 0:
            print(epoch)
            model.eval()
            steps.append(epoch)
            train_a =  correct / len(train_loader.dataset)
            train_accuracy.append(train_a.detach().numpy())
            train_loss.append(loss_train.detach().numpy()/count)
            t_loss = 0
            correct = 0
            num = 0
            for i_batch, (input, target) in enumerate(test_loader):
                num += 1
                if cuda:
                    input, target = input.cuda(), target.cuda()
                input, target = Variable(input), Variable(target)
                optimizer.zero_grad()
                output = model(input)
                predits = output.data.max(1)[1]
                correct += predits.eq(target.data).sum()
                t_loss += F.nll_loss(output, target)
            t_accuracy =  correct / len(test_loader.dataset)
            test_accuracy.append(t_accuracy.detach().numpy())
            test_loss.append(t_loss.detach().numpy()/num)
    graph(steps, train_accuracy, test_accuracy, train_loss, test_loss)
    return
                                                            


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT, help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                                            help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                                            help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                                            help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
