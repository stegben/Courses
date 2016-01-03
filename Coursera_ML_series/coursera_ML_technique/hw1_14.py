import numpy as np
from sklearn.svm import SVC


def read_data(filename, positive_class):
    with open(filename, "r") as f:
        X = []
        Y = []
        for row in f:
            data = [float(r) for r in row.rstrip().lstrip().split()]
            _y = 1 if data[0] == positive_class else 0
            _x = data[1:]

            Y.append(_y)
            X.append(_x)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

def main():
    trainX, trainY = read_data("train.txt", positive_class=0)
    testX, testY = read_data("test.txt", positive_class=0)

    model = SVC(C=0.01, 
                kernel='linear',
                verbose=True )

    model.fit(trainX, trainY)
    print(model.score(testX, testY))
    print(model.coef_)

if __name__ == "__main__":
    main()