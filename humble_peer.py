import scipy.io as sio
import numpy as np
import pickle
import math
import argparse

np.set_printoptions(precision=4, suppress=True)

def get_args():
    parser = argparse.ArgumentParser(description='Run Dynamic Grasping Experiment')
    args = parser.parse_args()

    # args.data_filepath = "kaggle_music.p"
    # args.data_filepath = "landmine.p"
    # args.data_filepath = "emails.p"
    args.data_filepath = "sentiment.p"


    args.sparse = False if args.data_filepath in ["landmine.p"] else True
    with open(args.data_filepath, "rb") as f:
        [args.X_train, args.Y_train, args.X_test, args.Y_test, args.k, args.fea] = pickle.load(f)
    if args.data_filepath in ["emails.p", "sentiments"]:
        # emails.p is in COO format
        args.X_train = args.X_train.tocsr()
        args.X_test = args.X_test.tocsr()
    args.run = 10
    return args

def test(w, k, X, Y, alg, sparse=False):
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

def get_std_error(a):
    """
    Get standard error, given samples.
    :param a: 1d array like
    """
    a = np.array(a)
    n = len(a)
    # return np.std(a, ddof=1)/np.sqrt(n)
    # Note now actually just get std deviation
    return np.std(a, ddof=1)

def print_summary(model, acc_list, query_count, err_count, total_count, run):
    """
    :param model: name of this model
    :param acc_list: list of run elements
    :param query_count: (k, run) ndarray
    :param err_count: (k, run) ndarray
    :param run: number of runs
    """
    accuracy = np.average(acc_list)
    query = np.sum(query_count) / run
    mistake_rates = np.sum(err_count, axis=0)/np.sum(total_count, axis=0)
    mistake_rate = np.average(mistake_rates)
    acc_std_err = get_std_error(acc_list)
    query_std_err = get_std_error(np.sum(query_count, axis=0))
    mistake_std_err = get_std_error(mistake_rates)

    print()
    print(model)
    print("accuracy: {:.4f}, acc_std_err: {:.4f}, query: {:.1f}, query_std_err: {:.1f}, mistakes: {:.4f}, mistakes_std_err: {:.4f}"\
          .format(accuracy, acc_std_err, query, query_std_err, mistake_rate, mistake_std_err))

def random(X, Y, X_test, Y_test, k, fea, query_limit='inf', sparse=False, run=10):
    query_limit = float(query_limit)
    query_count = np.zeros((k, run))
    total_count = np.zeros((k, run))
    err_count = np.zeros((k, run))
    acc = []

    for r in range(run):
        print("run: {}".format(r))
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
            z = p if np.sum(query_count, axis=0)[r] < query_limit else 0
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid][r] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count[tid][r] += 1
            total_count[tid][r] += 1

            if np.sum(query_count, axis=0)[r] >= query_limit:
                break
        acc.append(test(w, k, X_test, Y_test, "random", sparse))

    print_summary("random", acc, query_count, err_count, total_count, run)
    return acc
    
def indep(X, Y, X_test, Y_test, k, fea, query_limit='inf', sparse=False, run=10):
    query_limit = float(query_limit)
    query_count = np.zeros((k, run))
    total_count = np.zeros((k, run))
    err_count = np.zeros((k, run))
    b = 1.0 # controls confidence
    acc = []
    
    for r in range(run):
        print("run: {}".format(r))
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
            z = p if np.sum(query_count, axis=0)[r] < query_limit else 0
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid][r] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count[tid][r] += 1
            total_count[tid][r] += 1

            if np.sum(query_count, axis=0)[r] >= query_limit:
                break
        acc.append(test(w, k, X_test, Y_test, "independent", sparse))

    print_summary("indep", acc, query_count, err_count, total_count, run)
    return acc


def peer(X, Y, X_test, Y_test, k, fea, query_limit='inf', sparse=False, run=30, share=False):
    query_limit = float(query_limit)
    query_count = np.zeros((k, run)) # number of oracle queries
    total_count = np.zeros((k, run)) # number of total training examples
    err_count = np.zeros((k, run)) # number of error prediction
    peer_count = np.zeros((k, run)) # num of peer queries
    b = 1.0 # controls self confidence
    bq = 5.0 # controls peer confidence
    acc = []

    for r in range(run):
        print("run: {}".format(r))
        shuffle = np.random.permutation(X.shape[0])
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
                    peer_count[tid][r] += 1
                    y_hat = np.sign(fpeer)
                    if y_hat == 0:
                        y_hat = 1
            z = p * q
            if np.sum(query_count, axis=0)[r] >= query_limit:
                z = 0
            
            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid][r] += 1
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
                query_count[tid][r] += 1
            total_count[tid][r] += 1

            #### my modification
            # now only share data if true label queried
            if share:
                for j in range(k):
                    if tau[tid][j] >= tau[tid][tid] and j!=tid:
                        if yt != np.sign(f_t[j]):
                            w[j] = w[j] + z * yt * x

            if np.sum(query_count, axis=0)[r] >= query_limit:
                break
        acc.append(test(w, k, X_test, Y_test, "peer", sparse))

    model = "PEER+share" if share else "PEER"
    print_summary(model, acc, query_count, err_count, total_count, run)
    return acc

def committee(X, Y, X_test, Y_test, k, fea, query_limit='inf', sparse=False, run=30, C=0, share=False):

    # hyperparameters
    b = 1  # controls self confidence
    C = C  # controls how much task weight reduces based on loss

    query_limit = float(query_limit)
    query_count = np.zeros((k, run)) # number of oracle queries
    total_count = np.zeros((k, run)) # number of total training examples
    err_count = np.zeros((k, run)) # number of error prediction
    acc = []

    for r in range(run):
        print("run: {}".format(r))
        shuffle = np.random.permutation(X.shape[0])
        tau = np.ones((k, k))
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

            f_total = f_t.dot(tau[tid])

            y_hat = np.sign(f_total)
            if y_hat == 0:
                y_hat = -1
            uncertainty = b / (b + abs(f_total))
            z = np.random.binomial(1, uncertainty)
            if np.sum(query_count, axis=0)[r] >= query_limit:
                z = 0

            yt = Y[shuffle[i]]
            if yt == 0:
                yt = -1
            if yt != y_hat:
                err_count[tid][r] += 1
                w[tid] = w[tid] + yt * z * x
            if z == 1:
                query_count[tid][r] += 1

            f_t_sign = np.sign(f_t)
            for i in range(k):
                if f_t_sign[i] == 0:
                    f_t_sign[i] = -1

            if z == 1:
                l_t = np.fmin([2] * k, np.fmax([0] * k, 1 - f_t * yt)) # NOTE uppper bound the loss
                if sum(l_t) != 0:
                    l_t = l_t / sum(l_t)
                for tsk in range(k):
                    tau[tid][tsk] = tau[tid][tsk] * np.exp(- C * l_t[tsk])
                tau[tid] = tau[tid] * k / sum(tau[tid])
            total_count[tid][r] += 1

            ### my modification
            # now only share data if true label queried
            if share:
                for j in range(k):
                    if tau[tid][j] >= tau[tid][tid] and j!=tid:
                        if yt != f_t_sign[j]:
                            w[j] = w[j] + z * yt * x

            if np.sum(query_count, axis=0)[r] >= query_limit:
                break

        w = tau.dot(w)
        acc.append(test(w, k, X_test, Y_test, "peer", sparse))

    print_summary("committee", acc, query_count, err_count, total_count, run)
    return acc

def print_dataset_info(args):
    print("data set: {}".format(args.data_filepath))
    print("X_train.shape: {}".format(args.X_train.shape))
    print("Y_train.shape: {}".format(args.Y_train.shape))
    print("X_test.shape: {}".format(args.X_test.shape))
    print("Y_test.shape: {}".format(args.Y_test.shape))
    print("num of tasks: {}".format(args.k))
    print("num of training data per task: {}".format(args.X_train.shape[0]/k))
    print("num of test data per task: {}".format(args.X_test.shape[0] / k))
    print("num of features: {}".format(args.fea-1))

    num_positive_point = np.sum(args.Y_train)+np.sum(args.Y_test)
    num_data = args.X_train.shape[0] + args.X_test.shape[0]
    print("num of positive points: {} ({})".format(num_positive_point, num_positive_point/num_data))

# def test_with_limited_query(data_filepath, model, X_train, Y_train, X_test, Y_test, k, fea, sparse, run, share):
#     if data_filepath == "landmine.p":
#         query_ratio = np.linspace(0.01, 0.1, 10)
#     else:
#         query_ratio = np.linspace(0.03, 0.3, 10)
#     if model.__name__ in ['random', 'independent']
#     for q in query_ratio:
#         model()


if __name__ == "__main__":
    args = get_args()
    X_train, Y_train, X_test, Y_test, k, fea = args.X_train, args.Y_train, args.X_test, args.Y_test, args.k, args.fea
    sparse, run = args.sparse, args.run

    print_dataset_info(args)

    # random(X_train, Y_train, X_test, Y_test, k, fea, query_limit='inf', sparse=sparse, run=run)
    # indep(X_train, Y_train, X_test, Y_test, k, fea, query_limit='inf', sparse=sparse, run=run)
    # peer(X_train, Y_train, X_test, Y_test, k, fea, query_limit='inf', sparse=sparse, run=run)
    # peer(X_train, Y_train, X_test, Y_test, k, fea, query_limit='inf', sparse=sparse, run=run, share=True)
    # committee(X_train, Y_train, X_test, Y_test, k, fea, query_limit='inf', sparse=sparse, run=run, C=np.log(30), share=True)

    query_ratio = np.linspace(0.03, 0.3, 10)
    query_limit = query_ratio * X_train.shape[0]
    for q in query_limit:
        print(q)
        random(X_train, Y_train, X_test, Y_test, k, fea, query_limit=q, sparse=sparse, run=run)
        indep(X_train, Y_train, X_test, Y_test, k, fea, query_limit=q, sparse=sparse, run=run)
        peer(X_train, Y_train, X_test, Y_test, k, fea, query_limit=q, sparse=sparse, run=run)
        peer(X_train, Y_train, X_test, Y_test, k, fea, query_limit=q, sparse=sparse, run=run, share=True)
        committee(X_train, Y_train, X_test, Y_test, k, fea, query_limit=q, sparse=sparse, run=run, C=np.log(30), share=True)








