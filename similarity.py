import scipy.io as sio
import numpy as np
import pickle
from scipy.sparse import coo_matrix

np.set_printoptions(precision=2)

def similarity(X, Y, k, fea, sparse=False):
    center = np.zeros((k, fea - 1))
    count = [0 for i in range(k)]
    for i in range(X.shape[0]):
        if sparse:
            x = X.getrow(i).toarray()
            tid = int(x[0][0])
            x = x[0][1:]
        else:
            x = X[i][1:]
            tid = int(X[i][0])
        yt = Y[i]
        if yt == 1:
            center[tid] += x
            count[tid] += 1
    
    for i in range(k):
        center[i] /= count[i]
        center[i] /= np.linalg.norm(center[i])
    
    # average pairwise cosine similarity
    avg = 0.0
    for i in range(k):
        for j in range(k):
            if i <= j:
                continue
            avg += center[i].dot(center[j])
    avg /= (k * (k - 1)) / 2
    print(avg)

if __name__ == "__main__":
    # with open("landmine.p", "rb") as f:
    #     [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)

    # with open("landmine_unbalanced.p", "rb") as f:
    #     [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)

    with open("emails.p", "rb") as f:
        [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)

    print("X_train.shape", X_train.shape)
    print("Y_train.shape", Y_train.shape)
    print("X_test.shape", X_test.shape)
    print("Y_test.shape", Y_test.shape)
    print(k, fea)
    sparse = True
    similarity(X_train, Y_train, k, fea, sparse)
