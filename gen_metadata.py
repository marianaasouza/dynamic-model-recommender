import numpy as np
import os
import argparse
import pyhocon
import logging

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors

from folp import FOLP
from extract_metafeatures import extract_metafeatures


if __name__ == "__main__":

    root_dir = os.path.abspath(os.path.curdir)

    logging.basicConfig(
        filename=os.path.join(root_dir, 'gen_metadata.log'),
        level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Dynamic model type recommender')
    parser.add_argument('--config', type=str,
                        default=os.path.join(root_dir, 'experiment.conf'))
    args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file(args.config)

    dataset_names = config['data_setting.data_names']
    n_exec = int(config['data_setting.n_folds'])

    # Paths
    data_fpath = os.path.join(root_dir, config['data_setting.data_folder'])
    mdata_fpath = os.path.join(root_dir,
                               config['data_setting.metadata_folder'])

    init = 0
    end = len(dataset_names)
    idx_datasets = np.arange(init, end)

    # System parameters
    K = int(config['parameter_setting.K'])
    threshold = float(config['parameter_setting.threshold'])
    mfeat_names = config['parameter_setting.metafeatures']
    prtfl_names = config['parameter_setting.portfolio']

    dict_prtfl = {
        'Perceptron': Perceptron(),
        'DS': DecisionTreeClassifier(max_depth=1),
        'DT': DecisionTreeClassifier(),
        'LSVM': SVC(kernel='linear'),
        'GSVM': SVC(kernel='rbf')
    }

    prtfl = [dict_prtfl[mdl] for mdl in prtfl_names]

    n_mfeat = len(mfeat_names)
    n_prtfl = len(prtfl_names)

    # Generate meta-test sets
    logging.info('Generating meta-test sets')
    for i in idx_datasets:
        logging.info('Dataset: %s', dataset_names[i])
        for j in np.arange(0, n_exec):
            logging.info('Fold: %d', j)

            X_train = np.load(os.path.join(data_fpath, dataset_names[i] +
                                           '-f-' + str(j) + '-X-tra.npy'))
            X_test = np.load(os.path.join(data_fpath, dataset_names[i] +
                                          '-f-' + str(j) + '-X-tst.npy'))
            y_train = np.load(os.path.join(data_fpath, dataset_names[i] +
                                           '-f-' + str(j) + '-y-tra.npy'))
            y_test = np.load(os.path.join(data_fpath, dataset_names[i] +
                                          '-f-' + str(j) + '-y-tst.npy'))

            # Obtain the K-NN of each test sample over the training set
            nn = NearestNeighbors(n_neighbors=K).fit(X_train, y_train)
            nn_idx = nn.kneighbors(X_test, return_distance=False)

            X_mtest = np.zeros((X_test.shape[0], n_mfeat), dtype=float)
            y_mtest = np.zeros((X_test.shape[0], n_prtfl), dtype=int)
            y_mhit = np.zeros((X_test.shape[0], n_prtfl), dtype=int)

            clf = FOLP()
            clf.fit(X_train,y_train)

            # Generate meta-labels
            for m in np.arange(0,n_prtfl):
                clf.model = prtfl[m]
                y_scores = clf.predict_proba(X_test)[:, 1]
                y_out = clf.predict(X_test)
                mask_conf = np.logical_or(y_scores >= threshold,
                                          y_scores <= 1 - threshold)
                mask_corr = y_out == y_test
                y_mhit[mask_corr, m] = 1
                y_mtest[np.logical_and(mask_conf, mask_corr), m] = 1

            # Generate meta-features
            for n in np.arange(0,X_test.shape[0]):
                X_nn = X_train[nn_idx[n, :], :]
                y_nn = y_train[nn_idx[n, :]]
                X_mtest[n,:] = extract_metafeatures(X_nn, y_nn)

            if not os.path.exists(mdata_fpath):
                os.makedirs(mdata_fpath)

            np.save(os.path.join(mdata_fpath, dataset_names[i] + '-f-' +
                                 str(j) + '-X-mtst.npy'), X_mtest)

            np.save(os.path.join(mdata_fpath,
                                 dataset_names[i] + '-f-' + str(j) +
                                 '-y-mtst.npy'), y_mtest)
            np.save(os.path.join(mdata_fpath,
                                 dataset_names[i] + '-f-' + str(j) +
                                 '-y-mhit.npy'), y_mhit)

    # Generate meta-training sets

    logging.info('Generating meta-training sets')
    for i in idx_datasets:
        logging.info('Dataset: %s', dataset_names[i])

        train_datasets = [a for a in idx_datasets if a != i]

        X_mtrain = np.empty(shape=(0, n_mfeat), dtype=float)
        y_mtrain = np.empty(shape=(0, n_prtfl), dtype=int)

        for d in train_datasets:
            for j in np.arange(0, n_exec):
                X = np.load(os.path.join(mdata_fpath, dataset_names[d] +
                                         '-f-' + str(j) + '-X-mtst.npy'))
                y = np.load(os.path.join(mdata_fpath, dataset_names[d] +
                                         '-f-' + str(j) + '-y-mtst.npy'))

                lcard = np.sum(y,axis=1)
                mask_dist = np.logical_and(lcard > 0, lcard < n_prtfl)

                X_mtrain = np.append(X_mtrain, X[mask_dist, :], axis=0)
                y_mtrain = np.append(y_mtrain, y[mask_dist, :], axis=0)

        np.save(os.path.join(mdata_fpath, dataset_names[i] + '-X-mtra.npy'),
                X_mtrain)
        np.save(os.path.join(mdata_fpath, dataset_names[i] + '-y-mtra.npy'),
                y_mtrain)
