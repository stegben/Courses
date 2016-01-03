from __future__ import division
import sys
import numpy as np
import random


def gen_data(size=20, dim=1, upper=1, lower=-1, noise=0.2):
    X = np.random.uniform(low=lower, high=upper, size=(size, dim))
    Y = np.zeros(size)
    Y[np.where(X[:,0]>0)] = 1
    for ind in range(len(Y)):
        if random.random() < noise:
            Y[ind] = 0 if Y[ind] == 1 else 1
    return X, Y


class DecisionStump(object):
    def __init__(self, feat_dim=1, ):
        self.feat_dim = feat_dim
        self.theta = 0 # stump location
        self.s = 1 # stump direction
        self.i = 0 # stump using dimension


    def train(self, X, Y):
        best_error = 1
        best_theta = self.theta
        best_s = self.s
        best_i = self.i

        for i in range(self.feat_dim):
            # create candidate stump from all value in current dimension
            self.i = i
            feat = X[:, i]
            feat = np.sort(feat).tolist()
            feat = [feat[0]-1] + feat
            feat.append(feat[-1] + 1)
            stump_candidate_list = [ (feat[ind] + feat[ind+1]) / 2 for ind in range(len(feat)-1)]
            for s in [1, -1]:
                self.s = s
                for stump_candidate in stump_candidate_list:
                    self.theta = stump_candidate
                    cur_err = self.evaluate(X, Y)
                    if cur_err < best_error:
                        best_error = cur_err
                        best_theta = stump_candidate
                        best_s = s
                        best_i = i
        self.theta = best_theta
        self.s = best_s
        self.i = best_i
        return best_error

    def evaluate(self, X, Y):
        _y = self.predict(X)
        err = 1 - (_y == Y).sum()/len(Y)
        return err


    def predict(self, X):
        Y = np.zeros(X.shape[0])
        if self.s == 1:
            Y[np.where(X[:, self.i] > self.theta)] = 1
        elif self.s == -1:
            Y[np.where(X[:, self.i] < self.theta)] = 1
        return Y




def main():
    if len(sys.argv) > 1:
        epochs = int(sys.argv[1])
    else:
        epochs = 5000

    tr_result = []
    te_result = []
    for k in range(epochs):
        model = DecisionStump()

        Xtr, Ytr = gen_data()
        err_tr = model.train(Xtr, Ytr)
        tr_result.append(err_tr)

        Xte, Yte = gen_data()
        err_te = model.evaluate(Xte, Yte)
        te_result.append(err_te)

    print("Training Error:")
    print(np.mean(tr_result))

    print("Testing Error:")
    print(np.mean(te_result))


if __name__ == "__main__":
    main()