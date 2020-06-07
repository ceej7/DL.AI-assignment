import numpy as np
import matplotlib.pyplot as plt
from lr_utils import load_dataset

# gen labeled dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]
# print("训练集的数量: m_train = " + str(m_train))
# print("测试集的数量 : m_test = " + str(m_test))
# print("每张图片的宽/高 : num_px = " + str(num_px))
# print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print("训练集_图片的维数 : " + str(train_set_x_orig.shape))
# print("训练集_标签的维数 : " + str(train_set_y.shape))
# print("测试集_图片的维数: " + str(test_set_x_orig.shape))
# print("测试集_标签的维数: " + str(test_set_y.shape))
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# fixed number
m = m_train
n = num_px * num_px * 3
alpha = 0.009

# parameter
w = np.zeros((n, 1))
b = 0.5


def foward_propogation(_w, _X, _b):
    _Z = np.dot(_w.T, _X) + _b
    _Y = 1 / (1 + np.exp(-_Z))
    return _Z, _Y


# Training
X = train_set_x
Y = train_set_y
Costs = []
for i in range(1, 30000):
    Z, A = foward_propogation(w, X, b)
    dZ = A - Y
    dw = np.dot(X, dZ.T) / m
    db = np.sum(dZ) / m
    w = w - alpha * dw
    b = b - alpha * db

    Z_hat, Y_hat = foward_propogation(w, X, b)
    Cost = np.sum(np.dot(Y, np.diag(np.log(Y_hat).flat)) + np.dot(1 - Y, np.diag(np.log(1 - Y_hat).flat))) / (-m)
    Costs.append(Cost)

# Show Training Result
costs = np.squeeze(Costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(alpha))
plt.show()
plt.ion()
plt.pause(0.01)
input("Press Enter to Continue")
plt.close()

# Testing
X = test_set_x
Y = test_set_y
Z, A = foward_propogation(w, X, b)
for i in range(0, Y.shape[1]):
    print("Estimated [", i, "]: ", Y[0][i], "Guessed: ", A[0][i])
    plt.title(A[0][i])
    plt.imshow(test_set_x_orig[i])
    plt.ion()
    plt.pause(0.01)
    input("Press Enter to Continue")
    plt.close()