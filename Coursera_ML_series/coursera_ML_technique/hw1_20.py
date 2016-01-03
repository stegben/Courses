import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
import random


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
    record = []
    X, Y = read_data("train.txt", positive_class=0)
    print(X.shape, Y.shape)
    testX, testY = read_data("test.txt", positive_class=0)
    for itr in range(100):
        trainX, valX, trainY, valY = train_test_split(X, Y, test_size=1000,
                                                            random_state=random.randint(1,100000))
        best_gamma = None
        best_acc = 0.0
        for c in range(0,6):
            g = 10 ** c
            model = SVC(C=0.1, 
                        kernel='rbf',
                        gamma=g,
                        verbose=False )

            model.fit(trainX, trainY)
            val_score = model.score(valX, valY)
            if val_score > best_acc :
                best_acc = val_score
                best_gamma = g
        print(best_gamma)
        record.append(best_gamma)
    print(record)

if __name__ == "__main__":
    main()