import warnings
from typing import Union, Tuple
from core.abstract_models import _Decomposer, NotFittedException, InvalidNumOfComponentsException, \
    OneSamplePassedException, _Model
from core.kernels import Kernels, Kernel
import numpy as np


class KPCA(_Model, _Decomposer):
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0, degree: int = 3,
                 sigma: float = None, n_components: Union[int, float] = None, remove_zeros: bool = True):
        super().__init__()
        self.kernel = Kernel(kernel, alpha, coefficient, degree, sigma)
        self.n_components = n_components
        self.remove_zeros = remove_zeros
        self.alphas = None
        self.lambdas = None
        self.explained_var = None
        self._x_fit = None

    def _check_n_components(self, n_features: int) -> None:
        if self.n_components is None:
            self.n_components = n_features - 1
        elif self.n_components > n_features:
            self.n_components = n_features
        elif 1 <= self.n_components <= n_features:
            pass
        elif 0 < self.n_components < 1:
            self.n_components = self._pov_to_n_components()
        else:
            raise InvalidNumOfComponentsException('The number of components should be between 1 and {}, '
                                                  'or between (0, 1) for the pov, '
                                                  'in order to choose the number of components automatically.\n'
                                                  'Got {} instead.'
                                                  .format(n_features, self.n_components))

        self.explained_var = self.explained_var[:self.n_components]

    @staticmethod
    def _one_ns(shape: int) -> np.ndarray:
        return np.ones((shape, shape)) / shape

    @staticmethod
    def _center_matrix(kernel_matrix: np.ndarray) -> np.ndarray:
        if kernel_matrix.ndim == 1:
            return np.expand_dims(kernel_matrix, axis=0)

        m, n = kernel_matrix.shape

        one_m, one_n = KPCA._one_ns(m), KPCA._one_ns(n)

        return kernel_matrix - one_m.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_m, kernel_matrix, one_n])

    @staticmethod
    def _center_symmetric_matrix(kernel_matrix: np.ndarray) -> np.ndarray:
        n = kernel_matrix.shape[0]
        one_n = KPCA._one_ns(n)

        return kernel_matrix - one_n.dot(kernel_matrix) - kernel_matrix.dot(one_n) \
               + np.linalg.multi_dot([one_n, kernel_matrix, one_n])

    def _pov_to_n_components(self) -> int:
        pov = np.cumsum(self.explained_var)

        nearest_value_index = (np.abs(pov - self.n_components)).argmin()

        return nearest_value_index + 1

    def _clean_eigs(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        unwanted_indexes = np.where(eigenvalues < 0)

        if len(unwanted_indexes) > 0:
            warnings.warn('Negative eigenvalues where encountered!', RuntimeWarning)

        eigenvalues = np.delete(eigenvalues, unwanted_indexes)
        eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        if self.remove_zeros:
            unwanted_indexes = np.where(np.isclose(eigenvalues, 0))
            eigenvalues = np.delete(eigenvalues, unwanted_indexes)
            eigenvectors = np.delete(eigenvectors, unwanted_indexes, axis=1)

        eigenvectors = np.flip(eigenvectors, axis=1)
        eigenvalues = np.flip(eigenvalues)

        return eigenvalues, eigenvectors

    def fit(self, x: np.ndarray) -> np.ndarray:
        try:
            if x.shape[0] == 1:
                raise OneSamplePassedException('Cannot perform KPCA for 1 sample.')

            self._x_fit = x

            kernel_matrix = self.kernel.calc_matrix(self._x_fit)
            kernel_matrix = KPCA._center_symmetric_matrix(kernel_matrix)

            eigenvalues, eigenvectors = np.linalg.eigh(kernel_matrix)
            self.lambdas, self.alphas = self._clean_eigs(eigenvalues, eigenvectors)

            self.explained_var = self.lambdas / np.sum(self.lambdas)
            self._check_n_components(self._x_fit.shape[1])

            self.alphas = np.delete(self.alphas, np.s_[self.n_components:], axis=1)
            self.lambdas = np.delete(self.lambdas, np.s_[self.n_components:])
        except:
            pass

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._x_fit is None:
            raise NotFittedException('KPCA has not been fitted yet!')

        kernel_matrix = self.kernel.calc_matrix(self._x_fit, x)

        kernel_matrix = KPCA._center_matrix(kernel_matrix)

        return kernel_matrix.T.dot(self.alphas / np.sqrt(self.lambdas))

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        kernel_matrix = KPCA._center_matrix(self.fit(x))

        return kernel_matrix.T.dot(self.alphas / np.sqrt(self.lambdas))

    def get_params(self) -> dict:
        params = self.kernel.get_params()
        params['n_components'] = self._param_value(self.n_components)

        return params