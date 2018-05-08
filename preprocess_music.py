import operator
import numpy as np
import pickle
from scipy.sparse import csr_matrix

if __name__ == "__main__":
    song_to_tracks = {}
    train_tracks = set()
    top_users = set()
    feature = {}

    # song to tracks
    with open("song_to_tracks.txt") as f:
        for line in f:
            if len(line.strip().split()) != 2:
                continue
            [s, t] = line.strip().split()
            song_to_tracks[s] = t
    
    # train tracks 
    with open("train.txt") as f:
        for line in f:
            t = line.strip().split(",")[0]
            feature[t] = line.strip().split(",")[2:]
            train_tracks.add(t)
    
    print(len(train_tracks))
    print(len(song_to_tracks))
    
    # filter data
    filtered_data = []
    user_cnt = {}
    with open("triplets.txt") as f:
        for line in f:
            [u, s, n] = line.strip().split()
            if s in song_to_tracks and song_to_tracks[s] in train_tracks:
                filtered_data.append((u, song_to_tracks[s], n))
                if u in user_cnt:
                    user_cnt[u] += 1
                else:
                    user_cnt[u] = 0
    print(len(filtered_data))
    for key, value in sorted(user_cnt.items(), key=operator.itemgetter(1), reverse=True)[:100]:
        top_users.add(key)

    user_id = {}
    train_label = []
    test_label = []
    ntrain = 0
    ntest = 0
    row = []
    test_row = []
    col = []
    test_col = []
    data = []
    test_data = []

    for u, t, n in filtered_data:
        if u not in top_users:
            continue
        if u not in user_id:
            uid = len(user_id)
            user_id[u] = uid
        uid = user_id[u]
        y = int(int(n) > 1)

        z = np.random.binomial(1, 0.1)
        
        if z == 1:
            test_label.append(y)
            test_row.append(ntest)
            test_col.append(0)
            test_data.append(uid)
        else:
            train_label.append(y)
            row.append(ntrain)
            col.append(0)
            data.append(uid)

        for item in feature[t]:
            item = item.split(":")
            if z == 1:
                test_row.append(ntest)
                test_col.append(int(item[0]))
                test_data.append(float(item[1]))
            else:
                row.append(ntrain)
                col.append(int(item[0]))
                data.append(float(item[1]))
        if z == 1:
            ntest += 1
        else:
            ntrain += 1

    X_train = csr_matrix((data, (row, col)), shape=(ntrain, 5001))
    Y_train = np.array(train_label)
    fea = 5001
    X_test = csr_matrix((test_data, (test_row, test_col)), shape=(ntest, 5001))
    Y_test = np.array(test_label)
    k = len(user_id)
    
    with open("kaggle_music.p", "wb") as f:
        pickle.dump([X_train, Y_train, X_test, Y_test, k, fea], f)

    print("X.shape", X_train.shape)
    print("Y.shape", Y_train.shape)
    print(k, fea)

