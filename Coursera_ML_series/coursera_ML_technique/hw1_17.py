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
    for c in range(-3,2):
        C = 10**c

        trainX, trainY = read_data("train.txt", positive_class=0)
        testX, testY = read_data("test.txt", positive_class=0)

        model = SVC(C=C, 
                    kernel='rbf',
                    gamma=100,
                    verbose=False )

        model.fit(trainX, trainY)
        print(C)
        # print(model.score(trainX, trainY))
        print(model.score(testX, testY))
        print(len(model.support_vectors_))
        # print(sum(model.dual_coef_[0][len(model.dual_coef_[0])//2:]))

if __name__ == "__main__":
    main()