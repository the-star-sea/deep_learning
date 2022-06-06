from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
import matplotlib.pyplot as plt


def get_accu(outputs, batch_targets):

    _, predicted = torch.max(outputs.data, 1)
    total = batch_targets.size(0)
    hit = (predicted == batch_targets).sum().item()
    accuracy = 100.0 * hit / total

    return accuracy
def train(config):

    # Initialize the model that we are going to use
    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)  # fixme
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    steps = []
    losses = []
    accuracies = []
    lossu = 0.0
    steps=[]
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...
        optimizer.zero_grad()
        batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

        outputs = model(batch_inputs)
        loss = criterion(outputs, batch_targets)
        loss.backward()
        optimizer.step()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)

        # Add more code here ...

        lossu += loss.item()  # fixme

        if step % 50 == 0:
            steps.append(step)
            losses.append(lossu / 50)
            lossu = 0.0
            accu = get_accu(outputs, batch_targets)
            accuracies.append(accu)
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break


    print('Done training.')
    fig1 = plt.subplot(2, 1, 1)
    fig2 = plt.subplot(2, 1, 2)
    fig1.plot(steps, accuracies, c='red', label='accuracy')
    fig1.legend()
    fig2.plot(steps, losses , c='green', label='loss')
    fig2.legend()
    plt.show()
    return
if __name__ == "__main__":
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=15, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    config = parser.parse_args()
    # Train the model
    train(config)