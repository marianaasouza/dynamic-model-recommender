import numpy as np

from scipy.stats import mode

from sklearn.linear_model import Perceptron

from deslib.base import BaseDS
from sklearn.calibration import CalibratedClassifierCV

from deslib.util.instance_hardness import kdn_score

import copy

from sklearn.utils.validation import check_X_y
from sklearn.exceptions import NotFittedError


class FOLP(BaseDS):
    """
    Fitted Online Local Pool (FOLP).

    This technique dynamically generates and fits a pool of classifiers based
    on the local region each given
    query sample is located, if such region has any degree of class overlap.
    Otherwise, the technique uses the
    KNN rule for obtaining the query sample's label.

    Parameters
    ----------

    n_classifiers : int (default = 7)
             The size of the pool to be generated for each query instance.

    k : int (Default = 7)
        Number of neighbors used to obtain the Region of Competence (RoC).

    IH_rate : float (default = 0.0)
        Hardness threshold used to identify when to generate the
        local pool or not.

    ds_tech : str (default = 'ola')
        DCS technique to be coupled to the OLP.


    References
    ----------

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier
    selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin, Online local
    pool generation for
    dynamic classier selection, Pattern Recognition 85 (2019) 132-148.

    """

    def __init__(self, n_classifiers=5, k=7, IH_rate=0.0,
                 knne=True, model=Perceptron()):

        super(FOLP, self).__init__(k, IH_rate=IH_rate)

        self.name = 'TOLP'
        self.pool_classifiers = []
        self.knne = knne

        self.n_classifiers = n_classifiers
        self.k = k
        self.IH_rate = IH_rate
        self.model = model

    def fit(self, X, y):
        """
        Prepare the model by setting the KNN algorithm and
        calculates the information required to apply the OLP

         Parameters
        ----------
        X : matrix of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """
        check_X_y(X, y)
        self._set_dsel(X, y)
        self._set_region_of_competence_algorithm()
        self._fit_region_competence(X, y)

        # Set knne
        if self.n_classes > 2 or self.knne is None:
            self.knne = False

        # Calculate the KDN score of the training samples
        self.hardness, _ = kdn_score(X, y, self.k)

        return self

    def _set_dsel(self, X, y):
        """
        Get information about the structure of the data (e.g., n_classes,
        N_samples, classes)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        Returns
        -------
        self
        """
        self.DSEL_data = X
        self.DSEL_target = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        self.n_samples = self.DSEL_target.size

        return self

    def _validate_pool(self):
        """Check the n_estimator attribute."""
        if self.n_classifiers < 0:
            raise ValueError("n_classifiers must be greater than zero, "
                             "got {}.".format(self.n_classifiers))

    def _generate_local_pool(self, query):
        """
        Local pool generation.

        This procedure populates the "pool_classifiers" based on the
         query sample's neighborhood.
        Thus, for each query sample, a different pool is created.

        In each iteration, the training samples near the query sample are
        singled out and a
        subpool is generated using the
        Self-Generating Hyperplanes (SGH) method.
        Then, the DCS technique selects the best classifier in the generated
        subpool and it is added to the local pool.
        In the following iteration, the neighborhood is increased and
        another SGH-generated subpool is obtained
        over the new neighborhood, and again the DCS technique singles
        out the best in it, which is then added to the local pool.
        This process is repeated until the pool reaches "n_classifiers".

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample.

        Returns
        -------
        self

        References
        ----------

        M. A. Souza, G. D. Cavalcanti, R. M. Cruz, R. Sabourin,
        On the characterization of the
        oracle for dynamic classifier selection,
        in: International Joint Conference on Neural Networks,
        IEEE, 2017, pp. 332-339.
        """
        n_samples, _ = self.DSEL_data.shape

        self.pool_classifiers = []

        n_err = 0
        max_err = 2 * self.n_classifiers

        curr_k = self.k

        # Classifier count
        n = 0

        while n < self.n_classifiers and n_err < max_err:

            # subpool = SGH()

            included_samples = np.zeros((n_samples), int)

            if self.knne:
                # idx_neighb = self.obtain_knne(curr_k)
                idx_neighb = np.array([], dtype=int)

                # Obtain neighbors of each class individually
                for j in np.arange(0, self.n_classes):
                    # Obtain neighbors from the classes in the RoC
                    if np.any(self.classes[j] == self.DSEL_target[
                        self.neighbors[0][np.arange(0,
                                                    curr_k)]]):
                        nc = np.where(self.classes[j] == self.DSEL_target[
                            self.neighbors[0]])
                        idx_nc = self.neighbors[0][nc]
                        idx_nc = idx_nc[
                            np.arange(0, np.minimum(curr_k, len(idx_nc)))]
                        idx_neighb = np.concatenate((idx_neighb, idx_nc),
                                                    axis=0)

            else:
                idx_neighb = np.asarray(self.neighbors)[0][
                    np.arange(0, curr_k)]

            # Indicate participating instances in the training of the subpool
            included_samples[idx_neighb] = 1

            curr_classes = np.unique(self.DSEL_target[idx_neighb])

            # If there are +1 classes in the local region
            if len(curr_classes) > 1:
                p = self.model

                p.fit(self.DSEL_data[idx_neighb, :],
                      self.DSEL_target[idx_neighb])

                self.pool_classifiers.append(copy.deepcopy(p))

                n += 1

            # Increase neighborhood size
            curr_k += 2
            n_err += 1

        return self

    def select(self, query):
        """
        Obtains the votes of each classifier given a query sample.

        Parameters
        ----------
        query : array of shape = [n_features] containing the test sample

        Returns
        -------
        votes : array of shape = [len(pool_classifiers)] with the class
        yielded by each classifier in the pool

        """

        votes = np.zeros(len(self.pool_classifiers), dtype=int)
        for clf_idx, clf in enumerate(self.pool_classifiers):
            votes[clf_idx] = clf.predict(query)[0]

        return votes

    def classify_with_ds(self, query):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained
        by all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)

        # Predict query label
        if len(self.pool_classifiers) > 0:
            votes = self.select(query)
            predicted_label = mode(votes)[0]
        else:
            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]
            predicted_label = mode(self.DSEL_target[roc])[0]

        return predicted_label

    def _check_parameters(self):
        """
        Verifies if the input parameters are correct (k)
        raises an error if k < 1.
        """
        if self.k is not None:
            if not isinstance(self.k, int):
                raise TypeError("parameter k should be an integer")
            if self.k <= 1:
                raise ValueError("parameter k must be higher than 1."
                                 "input k is {} ".format(self.k))

        if self.safe_k is not None:
            if not isinstance(self.safe_k, int):
                raise TypeError("parameter safe_k should be an integer")
            if self.safe_k <= 1:
                raise ValueError("parameter safe_k must be higher than 1."
                                 "input safe_k is {} ".format(self.safe_k))

        if not isinstance(self.IH_rate, float):
            raise TypeError(
                "parameter IH_rate should be a float between [0.0, 0.5]")

        if 0 > self.IH_rate or self.IH_rate > 1:
            raise ValueError("Parameter IH_rate should be between [0.0, 1]."
                             "IH_rate = {}".format(self.IH_rate))

        self._validate_pool()

    def predict(self, X):
        """
        Predicts the class label for each sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if the DS model was trained
        self._check_is_fitted()

        n_samples = X.shape[0]
        predicted_labels = np.zeros(n_samples).astype(int)
        for index, instance in enumerate(X):

            instance = instance.reshape(1, -1)

            # proceeds with DS, calculates the region of competence of the query sample
            self.distances, self.neighbors = self._get_region_competence(
                instance, k=np.minimum(self.n_samples,
                                       self.n_classes * self.n_classifiers * self.k))

            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]

            # If all of its neighbors in the RoC have Instance hardness (IH)
            # below or equal to IH_rate, use KNN
            if np.all(self.hardness[np.asarray(roc)] <= self.IH_rate):
                y_neighbors = self.DSEL_target[roc]
                predicted_labels[index], _ = mode(y_neighbors)

            # Otherwise, generate the local pool for the query instance and
            # use DS for classification
            else:
                predicted_labels[index] = self.classify_with_ds(instance)

            self.neighbors = None
            self.distances = None

        return predicted_labels

    def classify_with_ds_proba(self, query):
        """
        Predicts the label of the corresponding query sample.

        The prediction is made by aggregating the votes obtained by all selected base classifiers.

        Parameters
        ----------
        query : array of shape = [n_features]
                The test sample

        Returns
        -------
        predicted_label : Prediction of the ensemble for the input query.
        """

        # Generate LP
        self._generate_local_pool(query)

        predicted_class_prob = np.zeros(len(self.classes))

        # Predict query label
        if len(self.pool_classifiers) > 0:

            proba = np.zeros((len(self.pool_classifiers), self.n_classes))
            for clf_idx, clf in enumerate(self.pool_classifiers):
                clf_calibrated = CalibratedClassifierCV(
                    base_estimator=clf, method='sigmoid',
                    cv='prefit').fit(
                    self.DSEL_data[
                    self.neighbors[0][np.arange(0, 4 * self.k)], :],
                    self.DSEL_target[
                        self.neighbors[0][np.arange(0, 4 * self.k)]])
                proba[clf_idx, :] = clf_calibrated.predict_proba(query)[0]

            predicted_class_prob = np.mean(proba, axis=0)

        else:
            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]
            y_neighbors = self.DSEL_target[roc]

            for c in np.arange(0, len(self.classes)):
                predicted_class_prob[c] = \
                    np.sum(y_neighbors == self.classes[c]) / len(roc)

        return predicted_class_prob

    def predict_proba(self, X):
        """
        Predicts the class label for each sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if the DS model was trained
        self._check_is_fitted()

        n_samples = X.shape[0]
        predicted_class_prob = np.zeros((n_samples, self.n_classes)).astype(
            float)
        for index, instance in enumerate(X):

            instance = instance.reshape(1, -1)

            # proceeds with DS, calculates the region of competence of the
            # query sample
            self.distances, self.neighbors = self._get_region_competence(
                instance, k=np.minimum(
                    self.n_samples,
                    self.n_classes * self.n_classifiers * self.k))

            nn = np.arange(0, self.k)
            roc = self.neighbors[0][nn]

            # If all of its neighbors in the RoC have Instance hardness (IH)
            # below or equal to IH_rate, use KNN
            if np.all(self.hardness[np.asarray(roc)] <= self.IH_rate):
                y_neighbors = self.DSEL_target[roc]
                for c in np.arange(0, len(self.classes)):
                    predicted_class_prob[index, c] = \
                        np.sum(y_neighbors == self.classes[c]) / len(roc)

            # Otherwise, generate the local pool for the query instance and
            # use DS for classification
            else:
                predicted_class_prob[index, :] = self.classify_with_ds_proba(
                    instance)

            self.neighbors = None
            self.distances = None

        return predicted_class_prob

    def _check_is_fitted(self):
        """ Verify if the dynamic selection algorithm was fitted. Raises an error if it is not fitted.

        Raises
        -------
        NotFittedError
            If the DS method is was not fitted, i.e., `self.roc_algorithm` or `self.processed_dsel`
             were not pre-processed.

        """
        if self.roc_algorithm_ is None:
            raise NotFittedError("DS method not fitted, "
                                 "call `fit` before exploiting the model.")
