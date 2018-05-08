import scipy.io as sio
import numpy as np
import pickle
import math

np.set_printoptions(precision=2)

run = 10 # number of random runs

def test(w, X, Y, alg, sparse=False):
    # return 0
    err_count = np.zeros((k, 1))
    for i in range(X.shape[0]):
        if sparse:
            x = X.getrow(i).toarray()
            tid = int(x[0][0])
            x = x[0][1:]
        else:
            x = X[i][1:]
            tid = int(X[i][0])
        x = np.concatenate((x, np.asarray([1])))
        x = x / np.linalg.norm(x)
        if alg == "pooled":
            y_hat = np.sign(w[0].dot(x))
        else:
            y_hat = np.sign(w[tid].dot(x))
        if y_hat == 0:
            y_hat = 1
        yt = Y[i]
        if yt == 0:
            yt = -1
        if yt != y_hat:
            err_count[tid] += 1
    return 1 - np.sum(err_count) / X.shape[0]
    # print("{:16s} accuracy {:.4f}".format(alg, 1 - np.sum(err_count) / X.shape[0]))

def random(X, Y, X_test, Y_test, k, fea, sparse=False):
    query_count = 0.0
    total_count = np.zeros((k, 1))
    err_count = np.zeros((k, 1))
    acc = []

    for r in range(run):
        shuffle = np.random.permutation(X.shape[0])
        w = np.zeros((k, fea))
        for i in range(X.shape[0]):
            if sparse:
                x = X.getrow(shuffle[i]).toarray()
                tid = int(x[0][0])
                x = x[0][1:]
            else:
                x = X[shuffle[i]][1:]
                tid = int(X[shuffle[i]][0])

            x = np.concatenate((x, np.asarray([1])))
            x = x / np.linalg.norm(x)
            
            y_hat = np.sign(w[tid].dot(x))
            if y_hat == 0:
                y_hat = 1
            p = np.random.binomial(1, 0.5)
            z = p
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count += 1
            total_count[tid] += 1
        acc.append(test(w, X_test, Y_test, "random", sparse))
    print("{:16s} query {:09.4f}\t mistakes {:.4f}\t accuracy {:.4f}".format("random",
        query_count / run, np.sum(err_count) / (X.shape[0] * run), sum(acc)/len(acc)))
    return acc
    
def indep(X, Y, X_test, Y_test, k, fea, sparse=False):
    query_count = 0.0
    total_count = np.zeros((k, 1))
    err_count = np.zeros((k, 1))
    b = 1.0 # controls confidence
    acc = []
    
    for r in range(run):
        shuffle = np.random.permutation(X.shape[0])
        w = np.zeros((k, fea))
        for i in range(X.shape[0]):
            if sparse:
                x = X.getrow(shuffle[i]).toarray()
                tid = int(x[0][0])
                x = x[0][1:]
            else:
                x = X[shuffle[i]][1:]
                tid = int(X[shuffle[i]][0])
            x = np.concatenate((x, np.asarray([1])))
            x = x / np.linalg.norm(x)
            
            f_t = w[tid].dot(x)
            y_hat = np.sign(f_t)
            if y_hat == 0:
                y_hat = 1

            uncertainty = b / (b + abs(f_t))
            p = np.random.binomial(1, uncertainty)
            z = p
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count += 1
            total_count[tid] += 1
        acc.append(test(w, X_test, Y_test, "independent", sparse))
    print("{:16s} query {:09.4f}\t mistakes {:.4f}\t accuracy {:.4f}".format("independent",
        query_count / run, np.sum(err_count) / (X.shape[0] * run), sum(acc)/len(acc)))
    return acc

def pooled(X, Y, X_test, Y_test, k, fea, sparse=False):
    query_count = 0.0
    total_count = np.zeros((k, 1))
    err_count = np.zeros((k, 1))
    b = 1.0 # controls confidence
    acc = []
    
    for r in range(run):
        shuffle = np.random.permutation(X.shape[0])
        w = np.zeros((1, fea))
        for i in range(X.shape[0]):
            if sparse:
                x = X.getrow(shuffle[i]).toarray()
                tid = int(x[0][0])
                x = x[0][1:]
            else:
                x = X[shuffle[i]][1:]
                tid = int(X[shuffle[i]][0])
            x = np.concatenate((x, np.asarray([1])))
            x = x / np.linalg.norm(x)
            
            f_t = w[0].dot(x)
            y_hat = np.sign(f_t)
            if y_hat == 0:
                y_hat = 1
            
            if math.isnan(f_t):
                continue

            uncertainty = b / (b + abs(f_t))
            p = np.random.binomial(1, uncertainty)
            z = p
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid] += 1
                w[0] = w[0] + yt * z * x
            if z == 1:
                query_count += 1
            total_count[tid] += 1
        acc.append(test(w, X_test, Y_test, "pooled", sparse))
    print("{:16s} query {:09.4f}\t mistakes {:.4f}\t accuracy {:.4f}".format("pooled",
        query_count / run, np.sum(err_count) / (X.shape[0] * run), sum(acc)/len(acc)))
    return acc

def peer(X, Y, X_test, Y_test, k, fea, query_limit=10**10, sparse=False, bq=6.0):
    query_count_tasks = []
    total_count = np.zeros((k, 1))
    err_count_tasks = []
    peer_count = 0.0
    b = 1.0 # controls self confidence
    bq = 5.0 # controls peer confidence
    acc = []

    for r in range(run):
        shuffle = np.random.permutation(X.shape[0])
        query_count = 0.0
        err_count = np.zeros((k, 1))
        tau = np.ones((k, k))/(k-1) - np.eye(k) * ((k-2) / (k-1))
        w = np.zeros((k, fea))
        for i in range(X.shape[0]):
            if sparse:
                x = X.getrow(shuffle[i]).toarray()
                tid = int(x[0][0])
                x = x[0][1:]
            else:
                x = X[shuffle[i]][1:]
                tid = int(X[shuffle[i]][0])
            x = np.concatenate((x, np.asarray([1])))
            x = x / np.linalg.norm(x)
            
            f_t = w.dot(x)
            y_hat = np.sign(f_t[tid])
            if y_hat == 0:
                y_hat = 1

            uncertainty = b / (b + abs(f_t[tid]))
            p = np.random.binomial(1, uncertainty)
            q = 0
            if p == 1: # uncertain, ask peers
                sel_tasks = [i for i in range(k) if i != tid]
                fpeer = f_t[sel_tasks].dot(tau[tid][sel_tasks].T)
                peer_uncertainty = bq / (bq + abs(fpeer))
                q = np.random.binomial(1, peer_uncertainty)
                if q == 0:
                    peer_count += 1
                    y_hat = np.sign(fpeer)
                    if y_hat == 0:
                        y_hat = 1
            z = p * q
            if query_count >= query_limit:
                z = 0
            
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid] += 1
                w[tid] = w[tid] + yt * z * x
                # update tau
                sel_tasks = [i for i in range(k) if i != tid]
                l_t = np.maximum(0, 1 - yt * f_t[sel_tasks])
                lambda_ = np.sum(l_t)
                if lambda_ == 0:
                    lambda_ = 1
                tau_h = np.multiply(tau[tid][sel_tasks], np.exp(-z * l_t / lambda_))
                tau[tid][sel_tasks] = tau_h / sum(tau_h)
            if p == 1 and q == 0:
                w[tid] += y_hat * x
            if z == 1:
                query_count += 1
            total_count[tid] += 1
            if query_count >= query_limit:
                break
        acc.append(test(w, X_test, Y_test, "peer", sparse))
        query_count_tasks.append(query_count)
        err_count_tasks.append(np.sum(err_count) / X.shape[0])
    print("{:16s} query {:09.4f}\t mistakes {:.4f}\t accuracy {:.4f}".format("peer",
        np.average(query_count_tasks), np.sum(err_count_tasks) / run, sum(acc)/len(acc)))
    # print(np.average(acc), np.std(acc))
    # print(np.average(query_count_tasks), np.std(query_count_tasks))
    # print(np.average(err_count_tasks), np.std(err_count_tasks))
    return acc

def new_peer(X, Y, X_test, Y_test, k, fea, mode, query_limit = 10**10, sparse=False):
    """
    mode is 'distance' or 'mistake'
    """
    def clip(x,y):
        return min(2, max(0, 1 - x * y))

    query_count_tasks = []
    total_count = np.zeros((k, 1))
    err_count_tasks = []
    peer_count = 0.0
    b = 1  # controls self confidence
    alpha = 1.1
    C = np.log(30)
    beta = 0.1
    acc = []

    for r in range(run):
        shuffle = np.random.permutation(X.shape[0])
        query_count = 0.0
        err_count = np.zeros((k, 1))
        tau = np.ones((k, k))
        w = np.zeros((k, fea))
        total_sum = np.zeros((k, fea))
        total_num = np.zeros((k,1))
        center = np.zeros((k, fea))
        
        for i in range(X.shape[0]):
            if sparse:
                x = X.getrow(shuffle[i]).toarray()
                tid = int(x[0][0])
                x = x[0][1:]
            else:
                x = X[shuffle[i]][1:]
                tid = int(X[shuffle[i]][0])
            x = np.concatenate((x, np.asarray([1])))
            x = x / np.linalg.norm(x)

            f_t = w.dot(x)
            if mode == 'distance':
                f_total = f_t.dot(tau[tid]) * k/sum(tau[tid])
            else:
                f_total = f_t.dot(tau[tid])
            y_hat = np.sign(f_total)
            if y_hat == 0:
                y_hat = -1
            uncertainty = b / (b + abs(f_total))
            z = np.random.binomial(1, uncertainty)
            if query_count >= query_limit:
                z = 0

            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count += 1

            if mode == 'distance':
                total_num[tid] = total_num[tid] + 1
                total_sum[tid] = total_sum[tid] + x
                new_center = total_sum[tid] / total_num[tid]
                center[tid] = new_center

                temp = center - new_center
                dist = np.linalg.norm(temp, axis = 1)
                weight = np.exp(C * dist)
                tau[tid] = weight
                tau[:,tid] = weight

            if mode == 'mistake':
                f_t_sign = np.sign(f_t)
                for i in range(k):
                    if f_t_sign[i] == 0:
                        f_t_sign[i] = -1

                if z == 1:
                    l_t = np.fmin([2] * k, np.fmax([0]*k, 1-f_t * yt))
                    l_t = l_t / sum(l_t)
                    for tsk in range(k):
                        tau[tid][tsk] = tau[tid][tsk] * np.exp(- C * l_t[tsk])
                    tau[tid] = tau[tid]*k/sum(tau[tid])

            total_count[tid] += 1
            if query_count >= query_limit:
                break
        w = tau.dot(w)
        acc.append(test(w, X_test, Y_test, "peer", sparse))
        query_count_tasks.append(query_count)
        err_count_tasks.append(np.sum(err_count) / X.shape[0])
    avg_acc = sum(acc) / len(acc)
    print("{:16s} query {:09.4f}\t mistakes {:.4f}\t accuracy {:.4f}".format("peer",
                                                                             np.average(query_count_tasks),
                                                                             np.sum(err_count_tasks) / run,
                                                                             avg_acc))
    # print(np.average(acc), np.std(acc))
    # print(np.average(query_count_tasks), np.std(query_count_tasks))
    # print(np.average(err_count_tasks), np.std(err_count_tasks))
    return acc

if __name__ == "__main__":
    # with open("landmine.p", "rb") as f:
    #     [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)

    # with open("emails.p", "rb") as f:
    #     [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)
    # X_train = X_train.tocsr()
    # X_test = X_test.tocsr()

    with open("kaggle_music.p", "rb") as f:
        [X_train, Y_train, X_test, Y_test, k, fea] = pickle.load(f)

    print("X_train.shape", X_train.shape)
    print("Y_train.shape", Y_train.shape)
    print("X_test.shape", X_test.shape)
    print("Y_test.shape", Y_test.shape)
    print(k, fea)

    # sparse = False
    sparse = True
    random(X_train, Y_train, X_test, Y_test, k, fea, sparse)
    indep(X_train, Y_train, X_test, Y_test, k, fea, sparse)
    pooled(X_train, Y_train, X_test, Y_test, k, fea, sparse)
    peer(X_train, Y_train, X_test, Y_test, k, fea, sparse=sparse)
    new_peer(X_train, Y_train, X_test, Y_test, k, fea, mode='mistake', sparse=sparse)

