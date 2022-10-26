import numpy as np
from typing import Union, Tuple
from core.abstract_models import _Decomposer, NotFittedException, InvalidNumOfComponentsException, \
    OneSamplePassedException, _Model


class LdaNotFeasibleException(Exception):
    pass


class LDA(_Model, _Decomposer):
    def __init__(self, n_components: Union[int, float] = None, remove_zeros: bool = True):
        super().__init__()

        self.n_components = n_components
        self.remove_zeros = remove_zeros
        self.explained_var = None
        self._labels = None
        self._labels_counts = None
        self._n_classes = None
        self._n_features = None
        self._w = None

    def _check_if_possible(self, x: np.ndarray) -> None:
        n_samples = x.shape[0]

        if n_samples < 2:
            raise OneSamplePassedException('Cannot perform Lda for 1 sample.')

        if n_samples < self._n_features:
            raise LdaNotFeasibleException('Lda is not feasible, '
                                          'if the number of components is less than the number of features.'
                                          'You seem to have {} components and {} features.'
                                          .format(n_samples, self._n_features))

    def __set_state(self, x: np.ndarray, y: np.ndarray) -> None:
        self._labels, self._labels_counts = np.unique(y, return_counts=True)
        self._n_classes = len(self._labels)
        self._n_features = x.shape[1]

    def _pov_to_n_components(self) -> int:
        pov = np.cumsum(self.explained_var)

        nearest_value_index = (np.abs(pov - self.n_components)).argmin()

        return nearest_value_index + 1

    def _check_n_components(self) -> None:
        if self.n_components is None:
            self.n_components = self._n_classes - 1
        elif self.n_components >= self._n_classes:
            self.n_components = self._n_classes - 1
        elif 1 <= self.n_components < self._n_classes:
            pass
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        else:
            raise InvalidNumOfComponentsException('The number of components should be between 1 and {}, '
                                                  'or between (0, 1) for the pov, '
                                                  'in order to choose the number of components automatically.\n'
                                                  'Got {} instead.'
                                                  .format(self._n_classes, self.n_components))

        self.explained_var = self.explained_var[:self.n_components]

    def _class_means(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        means = np.zeros((self._n_classes, self._n_features))

        for c, label in zip(range(self._n_classes), self._labels):
            means[c] += np.mean(x[y == label], axis=0, dtype=np.float64)

        return means

    def _sw(self, x: np.ndarray, y: np.ndarray, class_means: np.ndarray) -> np.ndarray:
        sw = np.zeros((self._n_features, self._n_features))

        for label_index, label in enumerate(self._labels):
            si = np.zeros((self._n_features, self._n_features))
            grouped_samples = x[y == label]
            n_grouped_samples = grouped_samples.shape[0]

            for sample, sample_index in zip(grouped_samples, range(n_grouped_samples)):
                diff = sample - class_means[label_index]
                diff = np.expand_dims(diff, axis=1)
                si += np.dot(diff, diff.T)
            sw += si

        return sw

    def _sb(self, class_means: np.ndarray, x_mean: np.ndarray) -> np.ndarray:
        sb = np.zeros((self._n_features, self._n_features))

        for mean_vec, count in zip(class_means, self._labels_counts):
            mean_vec_column = np.expand_dims(mean_vec, axis=1)
            diff = mean_vec_column - x_mean
            sb += count * np.dot(diff, diff.T)

        return sb

    def _clean_eigs(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unwanted_indexes = np.where(eigenvalues < 0)
        eigenvalues = np.delete(eigenvalues, unwanted_indexes)
        eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        if self.remove_zeros:
            unwanted_indexes = np.where(np.isclose(eigenvalues, 0))
            eigenvalues = np.delete(eigenvalues, unwanted_indexes)
            eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        eigenvectors = np.flip(eigenvectors, axis=1)
        eigenvalues = np.flip(eigenvalues)

        return eigenvalues, eigenvectors

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.__set_state(x, y)
        self._check_if_possible(x)

        x_mean = x.mean(axis=0, dtype=np.float64)
        class_means = self._class_means(x, y)

        sw = self._sw(x, y, class_means)
        sb = self._sb(class_means, np.expand_dims(x_mean, axis=1))
        sw_inv_sb = np.dot(np.linalg.inv(sw), sb)

        eigenvalues, eigenvectors = np.linalg.eigh(sw_inv_sb)
        eigenvalues, eigenvectors = self._clean_eigs(eigenvalues, eigenvectors)

        self.explained_var = np.divide(eigenvalues, np.sum(eigenvalues))
        self._check_n_components()

        self._w = np.delete(eigenvectors, np.s_[self.n_components:], axis=1)

        return self._w

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._w is None:
            raise NotFittedException('KPCA has not been fitted yet!')

        return np.dot(x, self._w)

    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(x, y)
        return self.transform(x)

    def get_params(self) -> dict:
        params = dict(n_components=self._param_value(self.n_components),
                      remove_zeros=self.remove_zeros)

        return params