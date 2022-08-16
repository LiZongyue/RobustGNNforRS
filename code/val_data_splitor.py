import world
import numpy as np
import os


def path_check():
    # Used to split training data into train and valid
    if not os.path.exists("../data/" + world.dataset + '/train_original.txt'):
        train_file = "../data/" + world.dataset + '/train.txt'

        trainUniqueUsers, trainItem, trainUser = [], [], []
        n_user = 0
        m_item = 0
        traindataSize = 0
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    m_item = max(m_item, max(items))
                    n_user = max(n_user, uid)
                    traindataSize += len(items)
        trainUser = np.array(trainUser)
        trainItem = np.array(trainItem)
        n = int(len(trainUser) * 0.1)
        index = np.random.choice(trainUser.shape[0], n, replace=False)

        valUser = trainUser[index]
        valItem = trainItem[index]

        trainUser = np.delete(trainUser, index, 0)
        trainItem = np.delete(trainItem, index, 0)

        train = np.array([[trainUser], [trainItem]]).squeeze().T
        val = np.array([[valUser], [valItem]]).squeeze().T

        os.rename(train_file, "../data/" + world.dataset + '/train_original.txt')
        np.savetxt("../data/" + world.dataset + '/train.txt', train, '%d')
        np.savetxt("../data/" + world.dataset + '/valid.txt', val, '%d')
