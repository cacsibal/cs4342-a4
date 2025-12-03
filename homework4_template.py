import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

IM_WIDTH = 48
NUM_INPUT = IM_WIDTH**2
NUM_HIDDEN = 20
NUM_OUTPUT = 1

def relu (z):
    return np.maximum(z, 0)

def relu_prime (z):
    return np.heaviside(z, 0) # relu'([0]) = [0]

def f_mse(yhat, y):
    return np.mean((yhat - y)**2) / 2

def forward_prop (x, y, W1, b1, W2, b2):
    z = W1 @ x + b1
    h = relu(z)
    yhat = W2 @ h + b2
    loss = f_mse(yhat, y)
    return loss, x, z, h, yhat

def back_prop (X, y, W1, b1, W2, b2):
    forward = forward_prop(X, y, W1, b1, W2, b2)
    yhat = forward[3]
    h = forward[2]

    gradW2 = (yhat - y) @ h
    gradb2 = yhat - y
    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-2, batchSize = 64, numEpochs = 1000):
    # TODO: finish me
    return W1, b1, W2, b2W

def show_weight_vectors (W1):
    # Show weight vectors in groups of 5.
    for i in range(NUM_HIDDEN//5):
        plt.imshow(np.hstack([ np.pad(np.reshape(W1[idx,:], [ IM_WIDTH, IM_WIDTH ]), 2, mode='constant') for idx in range(i*5, (i+1)*5) ]), cmap='gray'), plt.show()
    plt.show()

def loadData (which, mu = None):
    images = np.load("age_regression_X{}.npy".format(which)).reshape(-1, 48**2).T
    labels = np.load("age_regression_y{}.npy".format(which))

    if which == "tr":
        mu = np.mean(images)

    # TODO: you may wish to perform data augmentation (e.g., left-right flipping, adding Gaussian noise).

    return images - mu, labels, mu

def checkGradient():
    testW1 = np.load("testW1.npy")
    testb1 = np.load("testb1.npy")
    testW2 = np.load("testW2.npy")
    testb2 = np.load("testb2.npy")
    oneSampleX = np.load("oneSampleX.npy")
    oneSampley = np.load("oneSampley.npy")
    gradW1, gradb1, gradW2, gradb2 = back_prop(np.atleast_2d(oneSampleX).T, oneSampley, testW1, testb1, testW2, testb2)
    correctGradW1 = np.load("correctGradW1OnSample.npy")
    correctGradb1 = np.load("correctGradb1OnSample.npy")
    correctGradW2 = np.load("correctGradW2OnSample.npy")
    correctGradb2 = np.load("correctGradb2OnSample.npy")
    # The differences should all be <1e-5
    print(np.sum(np.abs(gradW1 - correctGradW1)))
    print(np.sum(np.abs(gradb1.flatten() - correctGradb1)))
    print(np.sum(np.abs(gradW2 - correctGradW2)))
    print(np.sum(np.abs(gradb2.flatten() - correctGradb2)))

if __name__ == "__main__":
    # Load data
    if "trainX" not in globals():
        trainX, trainY, mu = loadData("tr")
        testX, testY, _ = loadData("te", mu)

    # Check the gradient value for correctness.
    # Note: the gradients shown below assume 20 hidden units.
    checkGradient()

    # Initialize weights to reasonable random values
    W1 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_INPUT))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = np.mean(trainY)

    # Train NN
    W1, b1, W2, b2 = train(trainX, trainY, W1, b1, W2, b2, testX, testY)
