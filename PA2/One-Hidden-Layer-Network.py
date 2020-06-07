from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from testCases import *
import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

np.random.seed(1)

# data visual
X, Y = load_planar_dataset()
# print("X: ", X.shape)
# print("Y: ", Y.shape)
# plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)
# plt.show()
# plt.ion()
# plt.pause(0.01)
# input("Press Enter to Continue")
# plt.close()

# LR predict init
clf = sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T, Y.T)
# LR predict visual
# plot_decision_boundary(lambda x: clf.predict(x), X, np.squeeze(Y))
# plt.title("Logistic Regression")
# plt.show()
# plt.ion()
# plt.pause(0.01)
# input("Press Enter to Continue")
# plt.close()
# LR predict accuracy
LR_predictions = clf.predict(X.T)
print("逻辑回归的准确性： %d " % float(
    (np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) + "% " + "(正确标记的数据点所占的百分比)")


def layer_size(X, Y):
    n_x = X.shape[0]  # input layer
    n_h = 4  # hidden layer
    n_y = Y.shape[0]  # output layer
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parameters


def foward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    assert (W1.shape[1] == X.shape[0])
    assert (W1.shape[0] == b1.shape[0])
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    assert (A1.shape == (b1.shape[0], X.shape[1]))

    assert (W2.shape[1] == A1.shape[0])
    assert (W2.shape[0] == b2.shape[0])
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    assert (A2.shape == (b2.shape[0], X.shape[1]))

    caches = {
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }
    return caches


def compute_cost(Y_hat, Y):
    m = Y.shape[1]
    Cost = np.sum(np.dot(Y, np.diag(np.log(Y_hat).flat)) + np.dot(1 - Y, np.diag(np.log(1 - Y_hat).flat))) / (-m)
    assert (isinstance(Cost, float))
    return Cost


def back_propagation(parameters, caches, X, Y):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = caches["Z1"]
    A1 = caches["A1"]
    Z2 = caches["Z2"]
    A2 = caches["A2"]

    n_x = W1.shape[1]
    n_h = W1.shape[0]
    n_y = W2.shape[0]
    m = A2.shape[1]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis=1, keepdims = True)/m
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis=1, keepdims=True)/m
    assert (dW2.shape == W2.shape)
    assert (db2.shape == b2.shape)
    assert (dW1.shape == W1.shape)
    assert (db1.shape == b1.shape)

    grads = {
        "dW2": dW2,
        "db2": db2,
        "dW1": dW1,
        "db1": db1
    }
    return grads


def update_parameters(parameters, grads, learning_rate = 1.2):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def predict(parameters, X):
    c = foward_propagation(X, parameters)
    return np.round(c["A2"])


# one hidden layer neural network
num_iteration = 10000
n_x, n_h, n_y = layer_size(X, Y)
parameters = initialize_parameters(n_x, n_h, n_y)
for i in range(1, num_iteration):
    caches = foward_propagation(X, parameters)
    cost = compute_cost(caches["A2"], Y)
    grads = back_propagation(parameters, caches, X, Y)
    parameters = update_parameters(parameters, grads, learning_rate=0.6)
    print("Iteration: ", str(i), ", cost: ", cost)


# output result
prediction = predict(parameters, X)
plt.title("Decision Boundary for hidden layer size " + str(4))
plot_decision_boundary(lambda x: predict(parameters, x.T), X, np.squeeze(Y))
plt.show()
print("One-Hidden Layer的准确性： %d " % float(
    (np.dot(Y, prediction.T) + np.dot(1 - Y, 1 - prediction.T)) / float(Y.size) * 100) + "% " + "(正确标记的数据点所占的百分比)")
