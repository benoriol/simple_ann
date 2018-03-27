import numpy as np
import math

LEARNING_RATE = 0.2
# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps)

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    """
    m = y.shape[0]
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
Y = np.array([0,0,1,1]).reshape(4,1)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(10)

# initialize weights randomly with mean 0
#w0 = 2*np.random.random(3) - 1
w0 = np.random.randn(3,1)

for iter in range(100000):
    # for i in range(len(Y)):
    #     l0 = X[i]
    #     y = Y[i]
    #     print
    #     print 'w0: ', w0
    #     print 'l0: ', l0
    #     y_ = sigmoid(np.dot(w0, l0))
    #     print 'y: ', y
    #     print 'y_ ', y_
    #     dLdy_ =  - y/y_ + (1-y)/(1-y_)
    #     dy_da = y_ * (1 - y_)
    #     dLda = - y / (1 - y_)
    #     dadw = l0

    #     dLdw = dLdy_ * dy_da * dadw
    #     #dLdw = dLda * dadw

    #     w0 = w0 - LEARNING_RATE * dLdw

    Y_ = sigmoid(np.dot(X, w0))

    L = (1.0/4)*np.sum((-Y*np.log(Y_) - (1-Y)*np.log(1-Y_)))

    dLdy_ =  - Y/Y_ + (1-Y)/(1-Y_)
    dy_da = Y_ * (1 - Y_)
    dadw = X

    dLdw = (1.0/4)*np.dot(X.T, dLdy_*dy_da) 


    w0 = w0 - LEARNING_RATE*dLdw

    if iter%100 == 0:

        print
        print "*"*40
        print "Loss", L
        print "Y", Y
        print "Y_", Y_
        print "W0", w0
        print "*"*40
        print
        raw_input("...")

# print "Output After Training:"
# print y_
