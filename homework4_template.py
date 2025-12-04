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
    diff = np.atleast_1d(yhat - y)
    return np.mean(diff**2) / 2

def forward_prop(x, y, W1, b1, W2, b2):
    if x.ndim == 1:
        x = x[:, np.newaxis]

    b1_2d = b1.reshape(-1, 1) if b1.ndim == 1 else b1

    z = W1 @ x + b1_2d  # (NUM_HIDDEN, batchSize)
    h = relu(z)  # (NUM_HIDDEN, batchSize)

    yhat = (W2 @ h).ravel() + b2  # (batchSize,) or scalar
    loss = f_mse(yhat, y)

    return loss, x, z, h, yhat

def back_prop(X, y, W1, b1, W2, b2):
    loss, x, z, h, yhat = forward_prop(X, y, W1, b1, W2, b2)

    diff = np.atleast_1d(yhat - y)  # (batchSize,)

    if np.any(np.isnan(yhat)) or np.any(np.isinf(yhat)):
        print(f"yhat has NaN/Inf: min={np.min(yhat)}, max={np.max(yhat)}")
    if np.any(np.isnan(h)) or np.any(np.isinf(h)):
        print(f"h has NaN/Inf: min={np.min(h)}, max={np.max(h)}")

    g = (W2.T @ diff[np.newaxis, :]) * relu_prime(z) # (NUM_HIDDEN, batchSize)

    batchSize = x.shape[1]

    gradW1 = (g @ x.T) / batchSize  # (NUM_HIDDEN, NUM_INPUT)
    gradW2 = (diff @ h.T) / batchSize  # (1, NUM_HIDDEN)

    gradb1 = np.mean(g, axis=1)  # (NUM_HIDDEN,)
    gradb2 = np.mean(diff)  # scalar

    return gradW1, gradb1, gradW2, gradb2

def train (trainX, trainY, W1, b1, W2, b2, testX, testY, epsilon = 1e-2, batchSize = 64, numEpochs = 1000):
    for epoch in range(numEpochs):
        # shuffle
        idxs = np.arange(trainX.shape[1])
        np.random.shuffle(idxs)

        trainX = trainX[:, idxs]
        trainY = trainY[idxs]

        epoch_loss = 0
        num_batches = np.shape(trainX)[1] // batchSize
        for i in range(num_batches):
            X_batch = trainX[:, i * batchSize:(i + 1) * batchSize]
            y_batch = trainY[i * batchSize:(i + 1) * batchSize]

            gradW1, gradb1, gradW2, gradb2 = back_prop(X_batch, y_batch, W1, b1, W2, b2)

            W1 -= epsilon * gradW1
            b1 -= epsilon * gradb1
            W2 -= epsilon * gradW2
            b2 -= epsilon * gradb2

            if np.any(np.isnan(W1)) or np.any(np.isinf(W1)):
                print(f"Training stopped at epoch {epoch}: numerical instability detected")
                return W1, b1, W2, b2

            loss, _, _, _, _ = forward_prop(X_batch, y_batch, W1, b1, W2, b2)
            epoch_loss += loss

        training_loss = epoch_loss / num_batches
        testing_loss, _, _, _, _ = forward_prop(testX, testY, W1, b1, W2, b2)
        if epoch == 0 or (epoch + 1) % 100 == 0 or numEpochs - epoch <= 20:
            print(f"Epoch {epoch + 1}/{numEpochs}: Training half-MSE = {training_loss:.4f}, Testing half-MSE = {testing_loss:.4f}")

    return W1, b1, W2, b2

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
