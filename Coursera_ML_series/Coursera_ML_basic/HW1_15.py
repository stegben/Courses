import sys
from random import shuffle


filename = sys.argv[1]

f = open(filename, 'r')

iter = 2000
lr = 1

thr = 0
W = [0.]*4



def predict(X, W, thr):
    y = sum( [x*w for x,w in zip(X,W)] )
    y = y + thr
    if y > 0:
        return 1
    else:
        return -1

def main():

    tr = []
    for line in f:
        data = line.rstrip().split()
        data = [float(d) for d in data]
        tr.append(data)
        
    tmptr = tr
    
    n=0
    update_num_rec = []
    for k in range(iter):
        update_num = 0
        thr = 0
        W = [0.]*4
        shuffle(tmptr)
        while 1:
            er = 0
            for instance in tmptr:
                X = instance[:4]
                Y = instance[4]
                pre = predict(X, W, thr)
                if pre != Y:
                    er += 1
                    update_num += 1
                    for i in range(len(W)):
                        W[i] = W[i] + lr * Y * X[i]
                        thr = thr + lr * Y * 1
            if er == 0:
                break
        update_num_rec.append(update_num)
        print(n,update_num)
        n = n + 1
    print("Average Run Rounds:")
    print(sum(update_num_rec) / len(update_num_rec))
    
    
if __name__ == "__main__":
    main()