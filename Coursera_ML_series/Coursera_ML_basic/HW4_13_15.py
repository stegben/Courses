import numpy as np

epoch = 10
l2_list = range(-10, 3)

ftr = open("hw4_train.dat", 'r')
fte = open("hw4_test.dat", 'r')

X = []
Y = []

for line in ftr:
    data = [float(l) for l in line.rstrip().split()]
    data = [1] + data
    X.append(data[:-1])
    Y.append(data[-1])
X = np.array(X)
Y = np.array(Y)
    

Xte = []
Yte = []

for line in fte:
    data = [float(l) for l in line.rstrip().split()]
    data = [1] + data
    Xte.append(data[:-1])
    Yte.append(data[-1])
Xte = np.array(Xte)
Yte = np.array(Yte)


def error(y, pred, threshold=0):
    assert y.shape==pred.shape
    pred[pred>threshold] = 1
    pred[pred<=threshold] = -1
    return np.sum(y==pred)/y.shape[0]


for l2 in l2_list:
    Ein = []
    Eout = []
    reg = 10**l2
    for e in range(epoch):
        W = np.linalg.pinv(np.dot(X.T, X) + reg * np.identity(X.shape[1]), rcond=1e-20)
        W = np.dot(np.dot(W, X.T), Y)
        error_in = error(Y, np.dot(X, W))
        error_out = error(Yte, np.dot(Xte, W))
        # print(Y, np.dot(X, W))
        Ein.append(error_in)
        Eout.append(error_out)
    print(reg, sum(Ein)/len(Ein), sum(Eout)/len(Eout))


