import random
from matplotlib import pyplot as plt
import numpy as np
#task2
class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """
        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(2)
        
    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        label = np.sign(np.dot(self.weights, input) )
        return label
        
    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """
        for epoch in range(int(self.max_epochs)):
            shuffle(training_inputs,labels)
            for i in range(len(training_inputs)):
                if labels[i] * self.forward(training_inputs[i]) <= 0:
                    self.weights = self.weights + self.learning_rate * labels[i] * training_inputs[i]
def shuffle(X, y):
    randomlist = np.arange(len(y))
    np.random.shuffle(randomlist)
    return X[randomlist], y[randomlist]
def gausian(mean, cov):
    x, y = np.random.multivariate_normal(mean, cov, 200).T
    data = np.zeros((200, 2))
    for i in range(200):
        data[i][0] = x[i]
        data[i][1] = y[i]
    return data
if __name__=='__main__':
    # task1
    cov1=[[10,0],[0,10]]
    cov2=[[8,0],[0,8]]
    mean1=[1,1]
    mean2=[-1,-1]
    # print('Please input the mean1: ')
    # mean1 = [int(n) for n in input().split()]
    # print('Please input the cov1: ')
    # cov1[0] = [int(n) for n in input().split()]
    # cov1[1]=[int(n) for n in input().split()]
    # print('Please input the mean2: ')
    # mean2 = [int(n) for n in input().split()]
    # print('Please input the cov2: ')
    # cov2[0] = [int(n) for n in input().split()]
    # cov2[1]=[int(n) for n in input().split()]
    g1 = gausian(mean1, cov1)
    g2 = gausian(mean2, cov2)
    train1 = g1[0:160]
    test1 = g1[160:200]
    label1 = np.ones(160, dtype=np.int16)
    train2 = g2[0:160]
    test2 = g2[160:200]
    label2 = -label1
    plt.scatter(train1[:, 0], train1[:, 1],c='blue', label='positive data point')
    plt.scatter(train2[:, 0], train2[:, 1],c='green', label='negative data point')
    plt.title('Dataset generated from Guassian distribution')
    plt.show()
    plt.close()
    #task3
    model = Perceptron(320)
    train_data = np.append(train1, train2, 0)
    label_data=np.append(label1,label2, 0)
    model.train(train_data,label_data)
    correct = 0
    for i in range(40):
        if(model.forward(test1[i])==1):
            correct += 1
        if(model.forward(test2[i])==-1):
            correct+= 1
    print('Accuracy of test: ', 100*correct/80, '%')