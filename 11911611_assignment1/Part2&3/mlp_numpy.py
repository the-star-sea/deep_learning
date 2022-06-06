from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

softmax = SoftMax()
cross = CrossEntropy()


class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.linear_layers = []
        self.relu_layers = []
        in_size = n_inputs
        for i in range(len(n_hidden)):
            out_size = n_hidden[i]
            linear = Linear(in_size, out_size)
            self.linear_layers.append(linear)
            in_size = out_size
            self.relu_layers.append(ReLU())
        self.linear_layers.append(Linear(in_size, n_classes))

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """

        for i in range(len(self.n_hidden)):
            linear_out = self.linear_layers[i].forward(x)
            x = self.relu_layers[i].forward(linear_out)

        linear_out = self.linear_layers[-1].forward(x)

        out = SoftMax().forward(linear_out)

        return out

    def backward(self, dout, rate, grad_t):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        dx = SoftMax().backward(dout)
        dx = self.linear_layers[-1].backward(dx)
        if grad_t != 'BGD':
            self.linear_layers[-1].update(rate, 1)
        for i in range(len(self.relu_layers) - 1, -1, -1):
            dx = self.relu_layers[i].backward(dx)
            dx = self.linear_layers[i].backward(dx)
            if grad_t != 'BGD':
                self.linear_layers[i].update(rate, 1)
        return

    def update(self, rate, size):
        for linear in self.linear_layers:
            linear.update(rate, size)
        return
