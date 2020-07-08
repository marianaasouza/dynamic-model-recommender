import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
import collections
from math import log10
import random
from sklearn.svm import LinearSVC
from math import isnan

"""
Implementation of the data complexity measures used as meta-features 
in the proposed model-type dynamic recommender 
system.


References
----------

Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., & Ho, 
T. K. (2018). How Complex is your classification 
problem? A survey on measuring classification complexity. 
ArXiv:1808.03591 [Cs, Stat]. http://arxiv.org/abs/1808.03591

"""


def calculate_f3(X, y):
    """
    Calculates the Maximum Individual Feature Efficiency (F3) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    f3 : float
        Value of the F3 measure.
    """

    prop = np.ones(X.shape[1])
    classes = np.unique(y)
    f3 = 0.0

    if len(classes) > 1:
        c1 = X[y == classes[0], :]
        c2 = X[y == classes[1], :]
        for i in np.arange(0, X.shape[1]):
            min_c1 = np.min(c1[:, i])
            min_c2 = np.min(c2[:, i])
            max_c1 = np.max(c1[:, i])
            max_c2 = np.max(c2[:, i])

            mask_Xf = np.logical_and(X[:, i] <= np.min([max_c1, max_c2]),
                                     X[:, i] >= np.max([min_c1, min_c2]))
            prop[i] = np.sum(np.asarray(mask_Xf, dtype=int)) / X.shape[0]
        f3 = np.min(prop)

    return f3


def calculate_f4(X, y):
    """
    Calculates the Collective Feature Efficiency (F4) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    f4 : float
        Value of the F4 measure.
    """

    classes = np.unique(y)
    f4 = 0.0

    if len(classes) > 1:
        mask_X = np.ones(X.shape[0], dtype=bool)
        mask_f = np.ones(X.shape[1], dtype=bool)

        while np.any(mask_X) and np.any(mask_f):

            c1 = X[np.logical_and(y == classes[0], mask_X), :]
            c2 = X[np.logical_and(y == classes[1], mask_X), :]

            if c1.size == 0 or c2.size == 0:
                break

            else:
                prop = np.ones(X.shape[1]) + 0.5
                for i in np.arange(0, X.shape[1]):
                    if mask_f[i]:
                        min_c1 = np.min(c1[:, i])
                        min_c2 = np.min(c2[:, i])
                        max_c1 = np.max(c1[:, i])
                        max_c2 = np.max(c2[:, i])

                        mask_Xf = np.logical_and(
                            np.logical_and(X[:, i] <= np.min([max_c1, max_c2]),
                                           X[:, i] >= np.max(
                                               [min_c1, min_c2])), mask_X)
                        prop[i] = np.sum(np.asarray(mask_Xf, dtype=int)) / \
                                  np.sum(np.asarray(mask_X, dtype=int))

                idx_f = np.argmin(prop)
                mask_f[idx_f] = False
                mask_X = np.logical_and(
                    np.logical_and(X[:, idx_f] <= np.min(
                        [np.max(c1[:, idx_f]), np.max(c2[:, idx_f])]),
                                   X[:, idx_f] >= np.max([np.min(c1[:, idx_f]),
                                                          np.min(
                                                              c2[:, idx_f])])),
                    mask_X)

        f4 = np.sum(np.asarray(mask_X, dtype=int)) / X.shape[0]

    return f4


def calculate_n1(X, y):
    """
    Calculates the Fraction of Borderline Points (N1) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    n1 : float
        Value of the N1 measure.
    """

    n1 = 0.0
    classes = np.unique(y)

    if len(classes) > 1:
        class_1 = np.argwhere(y == classes[0])
        class_2 = np.argwhere(y == classes[1])
        enemies = np.array(np.meshgrid(class_1, class_2)).T.reshape(-1, 2)
        m_dist = squareform(pdist(X, metric='cityblock'))
        mst = minimum_spanning_tree(m_dist)

        stree = []
        for i in range(len(mst.nonzero()[0])):
            stree.append((mst.nonzero()[0][i], mst.nonzero()[1][i]))

        stree_set = [frozenset(w) for w in stree]
        enemies_set = [frozenset(w) for w in enemies]

        inters = set(stree_set) & set(enemies_set)

        pairs_list = [set(a) for a in inters]

        bord_samples = set.union(*map(set, pairs_list))

        n1 = len(bord_samples) / len(y)

    return n1


def calculate_n2(X, y):
    """
    Calculates the Ratio of Intra/Extra Class Nearest Neighbor Distance
    (N2) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    n2 : float
        Value of the N2 measure.
    """

    classes = np.unique(y)
    n2 = 0.0

    if len(classes) > 1:

        nbrs = NearestNeighbors(n_neighbors=X.shape[0], algorithm='kd_tree',
                                metric='cityblock').fit(X)
        _, idx = nbrs.kneighbors(X)

        nearest_enemies = np.zeros((y.shape), int)
        nearest_friends = np.zeros((y.shape), int)
        for i in np.arange(0, len(y)):
            diff_class = y[idx[i]] != y[i]
            nearest_enemy = np.argwhere(diff_class != 0)
            nearest_enemies[i] = idx[i, nearest_enemy[0][0]]

            equal_class = y[idx[i]] == y[i]
            possible_values = np.logical_and(equal_class != 0, idx[i] != i)
            nearest_friend = np.argwhere(possible_values)
            if np.any(nearest_friend):
                nearest_friends[i] = idx[i, nearest_friend[0][0]]
            else:
                nearest_friends[i] = idx[
                    i, nearest_enemy[len(nearest_enemy) - 1][0]]

        D = squareform(pdist(X, metric='cityblock'))
        intra_extra = np.nansum(D[np.arange(0, len(y)), nearest_friends]) / \
                      np.nansum(D[np.arange(0, len(y)), nearest_enemies])
        n2 = intra_extra / (1 + intra_extra)

    if isnan(n2):
        n2 = 1.0

    return n2


def calculate_n3(X, y):
    """
    Calculates the Error Rate of the Nearest Neighbor Classifier (N3) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    n3 : float
        Value of the N3 measure.
    """

    classes = np.unique(y)
    n3 = 0.0

    if len(classes) > 1:
        n_neighb = 1

        loo = LeaveOneOut()
        loo.get_n_splits(X)
        knn = KNeighborsClassifier(n_neighbors=n_neighb, metric='cityblock')

        for train_index, test_index in loo.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            knn.fit(X_train, y_train)
            hit_miss = knn.score(X_test, y_test)
            n3 = n3 + hit_miss
    else:
        n3 = len(y)

    return 1 - n3 / len(y)


def calculate_lsc(X, y):
    """
    Calculates the Local Set Average Cardinality (LSC) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    lsc : float
        Value of the LSC measure.
    """

    classes = np.unique(y)

    lsc = 0.0

    if len(classes) > 1:

        nbrs = NearestNeighbors(n_neighbors=X.shape[0], algorithm='kd_tree',
                                metric='cityblock').fit(X)
        D, idx = nbrs.kneighbors(X)

        n_ls = np.zeros((y.shape), int)

        for i in np.arange(0, len(y)):
            diff_class = y[idx[i]] != y[i]
            nearest_enemy = np.argwhere(diff_class != 0)

            n_ls[i] = nearest_enemy[0][0]

        lsc = 1 - np.sum(n_ls) / (len(y) ** 2)

    return lsc


def calculate_c2(y):
    """
    Calculates the Imbalance ratio (C2) measure.

    Parameters
    ----------

    y : array of shape = [n_samples]
        Class labels of the data.

    Returns
    -------
    c2 : float
        Value of the C2 measure.
    """

    counter = collections.Counter(y)
    freq_class = list(counter.values())
    if len(freq_class) > 1:
        c2 = max(freq_class) / min(freq_class)
    else:
        c2 = float(len(y))
    return c2


def calculate_c1(y):
    """
    Calculates the Entropy of class proportions (C1) measure.

    Parameters
    ----------

    y : array of shape = [n_samples]
        Class labels of the data.

    Returns
    -------
    c1 : float
        Value of the C1 measure.
    """

    c1 = 0.0
    counter = collections.Counter(y)
    classes = list(counter.keys())
    if len(classes) > 1:
        freq_class = list(counter.values())

        for i in np.arange(0, len(classes)):
            p = (freq_class[i] / len(y))
            c1 = c1 + p * log10(p)

        c1 = (-1 / (log10(len(classes)))) * c1

    return c1


def calculate_density(X, y):
    """
    Calculates the Average density of the network (Density) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    density : float
        Value of the Density measure.
    """

    n_samples = len(y)
    classes = np.unique(y)
    m_dist = np.asarray(squareform(pdist(X, metric='cityblock')))

    m_dist = m_dist - np.identity(n_samples)

    perc = 0.15
    min_dist = np.min(m_dist[m_dist >= 0])
    max_dist = np.max(m_dist[m_dist >= 0])

    rng = max_dist - min_dist
    epsilon = perc * rng + min_dist

    if len(classes) > 1:
        class_1 = np.argwhere(y == 0)
        class_2 = np.argwhere(y == 1)
        enemies = np.array(np.meshgrid(class_1, class_2)).T.reshape(-1, 2)
        for e in enemies:
            m_dist[e[0], e[1]] = -1

    idx_disc = np.argwhere(np.logical_and(m_dist > 0, m_dist < epsilon))

    density = 1 - len(idx_disc) / (n_samples * (n_samples - 1))

    return density


def calculate_l2(X, y):
    """
    Calculates the Error Rate of Linear Classifier (L2) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    l2 : float
        Value of the L2 measure.
    """

    classes = np.unique(y)

    l2 = 0.0

    if len(classes) > 1:
        svm = LinearSVC(max_iter=2000)
        svm.fit(X, y)
        l2 = 1 - svm.score(X, y)

    return l2


def calculate_n4(X, y):
    """
    Calculates the Non-Linearity of the Nearest Neighbor Classifier
    (N4) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    n4 : float
        Value of the N4 measure.
    """

    counter = collections.Counter(y)
    classes = list(counter.keys())
    n4 = 0.0
    if len(classes) > 1:
        freq_class = list(counter.values())

        X_test = np.zeros((X.shape))
        y_test = np.zeros((y.shape))
        idx_test = 0
        for i in np.arange(0, len(classes)):
            if freq_class[i] > 1:
                for j in np.arange(0, freq_class[i]):
                    curr_class = np.argwhere(y == classes[i]).T[0]

                    friends = [(a, b) for a in curr_class for b in curr_class
                               if a != b]

                    idx = random.randint(0, len(friends) - 1)
                    pos_curr = random.random()
                    pos_opp = 1 - pos_curr
                    X_test[idx_test, :] = (
                            pos_curr * X[friends[idx][0], :] +
                            pos_opp * X[friends[idx][1], :])
                    y_test[idx_test] = classes[i]
                    idx_test += 1

            else:
                curr_class = np.argwhere(y == classes[i]).T[0]
                opp_class = np.argwhere(y != classes[i]).T[0]
                enemies = [(a, b) for a in curr_class for b in opp_class]
                idx = random.randint(0, len(enemies) - 1)
                pos_curr = random.random()
                pos_opp = 1 - pos_curr
                X_test[idx_test, :] = (
                        pos_curr * X[enemies[idx][0], :] +
                        pos_opp * X[enemies[idx][1], :])
                y_test[idx_test] = classes[i]
                idx_test += 1

        y_test = np.asarray(y_test, dtype=int)
        knn = KNeighborsClassifier(n_neighbors=1, metric='cityblock')
        knn.fit(X, y)
        n4 = 1 - knn.score(X_test, y_test)

    return n4


def calculate_l3(X, y):
    """
    Calculates the Non-Linearity of a Linear Classifier (L3) measure.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    l3 : float
        Value of the L3 measure.
    """

    counter = collections.Counter(y)
    classes = list(counter.keys())
    l3 = 0.0

    if len(classes) > 1:
        freq_class = list(counter.values())

        X_test = np.zeros((X.shape))
        y_test = np.zeros((y.shape))
        idx_test = 0
        for i in np.arange(0, len(classes)):
            if freq_class[i] > 1:
                for j in np.arange(0, freq_class[i]):
                    curr_class = np.argwhere(y == classes[i]).T[0]

                    friends = [(a, b) for a in curr_class for b in curr_class
                               if a != b]

                    idx = random.randint(0, len(friends) - 1)
                    pos_curr = random.random()
                    pos_opp = 1 - pos_curr
                    X_test[idx_test, :] = (
                            pos_curr * X[friends[idx][0], :] +
                            pos_opp * X[friends[idx][1], :])
                    y_test[idx_test] = classes[i]
                    idx_test += 1

            else:
                curr_class = np.argwhere(y == classes[i]).T[0]
                opp_class = np.argwhere(y != classes[i]).T[0]
                enemies = [(a, b) for a in curr_class for b in opp_class]
                idx = random.randint(0, len(enemies) - 1)
                pos_curr = random.random()
                pos_opp = 1 - pos_curr
                X_test[idx_test, :] = (
                        pos_curr * X[enemies[idx][0], :] +
                        pos_opp * X[enemies[idx][1], :])
                y_test[idx_test] = classes[i]
                idx_test += 1

        y_test = np.asarray(y_test, dtype=int)
        svm = LinearSVC(max_iter=2000)
        svm.fit(X, y)

        l3 = 1 - svm.score(X_test, y_test)

    return l3


def extract_metafeatures(X, y):
    """
    Extracts the meta-features comprised of the 12 data complexity measures.

    Parameters
    ----------
    X : array of shape = [n_samples, n_features]
        Predictive attributes of the data.

    y : array of shape = [n_samples]
        Class labels of each example in X.

    Returns
    -------
    mfeat : array of shape = [n_metafeatures]
        Meta-feature vector.
    """

    # Feature-based measures
    f3 = calculate_f3(X, y)
    f4 = calculate_f4(X, y)

    # Linearity-based measures
    l2 = calculate_l2(X, y)
    l3 = calculate_l3(X, y)

    # Neighborhood-based measures
    n1 = calculate_n1(X, y)
    n2 = calculate_n2(X, y)
    n3 = calculate_n3(X, y)
    n4 = calculate_n4(X, y)
    lsc = calculate_lsc(X, y)

    # Network-based measures
    d = calculate_density(X, y)

    # Class imbalance-based measures
    c1 = calculate_c1(y)
    c2 = calculate_c2(y)

    # Meta-feature vector
    mfeat = [f3, f4, l2, l3, n1, n2, n3, n4, lsc, d, c1, c2]

    return np.asarray(mfeat)
