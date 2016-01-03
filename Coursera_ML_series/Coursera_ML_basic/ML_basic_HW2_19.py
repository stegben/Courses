from __future__ import division
import sys
import numpy as np



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
                print(best_i, best_theta, best_s, best_error)
        self.theta = best_theta
        self.s = best_s
        self.i = best_i
        return best_error

    def evaluate(self, X, Y):
        _y = self.predict(X)
        err = 1 - (_y == Y).sum()/len(Y)
        return err


    def predict(self, X):
        Y = np.ones(X.shape[0])
        if self.s == 1:
            Y[np.where(X[:, self.i] < self.theta)] = -1
        elif self.s == -1:
            Y[np.where(X[:, self.i] > self.theta)] = -1
        return Y


    def print_model(self):
        print("Stump: {}".format(self.theta))
        print("Used dimension: {}".format(self.i))
        print("Direction of stump: {}".format(self.s))



def gen_data_from_file(filename):
    f = open(filename, 'r')
    X = []
    Y = []
    for line in f:
        data = line.rstrip().split()
        x_ = [float(d) for d in data[:-1]]
        y_ = int(data[-1])
        X.append(x_)
        Y.append(y_)
    return np.array(X), np.array(Y)


def main():
    trfname = sys.argv[1]
    tefname = sys.argv[2]

    Xtr, Ytr = gen_data_from_file(trfname)
    Xte, Yte = gen_data_from_file(tefname)
    print(Xtr)
    assert Xtr.shape[1] == Xte.shape[1]

    model = DecisionStump(feat_dim=Xtr.shape[1])
    

    tr_err = model.train(Xtr, Ytr)
    te_err = model.evaluate(Xte, Yte)

    model.print_model()

    print(tr_err, te_err)



if __name__ == "__main__":
    main()