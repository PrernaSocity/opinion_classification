import numpy as np
from enum import Enum, auto
from core.abstract_models import _Model
from scipy.spatial.distance import pdist, squareform, cdist


class Kernels(Enum):
    LINEAR = auto()
    POLYNOMIAL = auto()
    RBF = auto()
    MIN = auto()


class InvalidDimensionsException(Exception):
    pass


class InvalidKernelException(Exception):
    pass


class IncompatibleShapesException(Exception):
    pass


class Kernel(_Model):
    def __init__(self, kernel: Kernels = Kernels.RBF, alpha: float = None, coefficient: float = 0, degree: int = 3,
                 sigma: float = None):
        super().__init__()
        self._kernel = kernel
        self._alpha = alpha
        self._coefficient = coefficient
        self._degree = degree
        self._sigma = sigma

    @staticmethod
    def _array_dim_check(x: np.ndarray) -> None:
        if x.ndim == 1:
            raise InvalidDimensionsException('Input arrays should be 2 dimensional.\n'
                                             'Got {} instead.\n'
                                             'You could use np.expand_dims(array, axis=0), '
                                             'in order to convert a 1D array with one sample to 2D\n'
                                             'or np.expand_dims(array, axis=1), '
                                             'in order to convert a 1D array with one feature to 2D.'
                                             .format(x.shape))
        if x.ndim != 2:
            raise InvalidDimensionsException('Input arrays should be 2 dimensional.\n'
                                             'Got {} instead.'
                                             .format(x.shape))

    @staticmethod
    def _arrays_check(x: np.ndarray, y: np.ndarray) -> None:
        Kernel._array_dim_check(x)

        if y is not None:
            Kernel._array_dim_check(y)
            if x.shape[1] != y.shape[1]:
                raise IncompatibleShapesException(
                    'Arrays column size should be the same. Got {} and {} instead'.format(x.shape[1], y.shape[1]))

    @staticmethod
    def _linear_kernel(x: np.ndarray, y: np.ndarray, coefficient: float) -> np.ndarray:
        if y is not None:
            dots = np.dot(x, y.T)

        else:
            n_samples = x.shape[0]

            dots = np.zeros((n_samples, n_samples))

            for i in range(n_samples):
                for j in range(n_samples):
                    dots[i, j] = np.dot(x[i, :].T, x[j, :])

        return dots + coefficient

    @staticmethod
    def _poly_kernel(x: np.ndarray, y: np.ndarray, alpha: float, coefficient: float, degree: int) -> np.ndarray:
        if y is not None:
            dots = np.dot(x, y.T)

        else:
            n_samples = x.shape[0]

            dots = np.zeros((n_samples, n_samples))

            for i in range(n_samples):
                for j in range(n_samples):
                    dots[i, j] = np.dot(x[i, :].T, x[j, :])

        return np.power(alpha * dots + coefficient, degree)

    @staticmethod
    def _rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:

        if y is not None:
            dists = cdist(x, y, 'sqeuclidean')

        else:
            dists = pdist(x, 'sqeuclidean')
            dists = squareform(dists)

        gamma = np.divide(1, np.multiply(2, np.square(sigma)))

        return np.exp(-gamma * dists)

    @staticmethod
    def _min_kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x_n_samples = x.shape[0]

        if y is not None:
            y_n_samples = y.shape[0]

            sums = np.zeros((y_n_samples, x_n_samples))
            for i in range(y_n_samples):
                for j in range(x_n_samples):
                    sums[i, j] = np.sum(np.minimum(x[i, :], y[j, :]))

        else:
            x_n_samples = x_n_samples

            sums = np.zeros((x_n_samples, x_n_samples))

            for i in range(x_n_samples):
                for j in range(x_n_samples):
                    sums[i, j] = np.sum(np.minimum(x[i, :], x[j, :]))

        return sums

    def calc_matrix(self, x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        Kernel._arrays_check(x, y)

        n_features = x.shape[1]

        if self._alpha is None:
            self._alpha = 1 / n_features

        if self._sigma is None:
            self._sigma = np.sqrt(n_features / 2)

        if self._kernel == Kernels.LINEAR:
            return Kernel._linear_kernel(x, y, self._coefficient)
        elif self._kernel == Kernels.POLYNOMIAL:
            return Kernel._poly_kernel(x, y, self._alpha, self._coefficient, self._degree)
        elif self._kernel == Kernels.RBF:
            return Kernel._rbf_kernel(x, y, self._sigma)
        elif self._kernel == Kernels.MIN:
            return Kernel._min_kernel(x, y)
        else:
            raise InvalidKernelException('Please choose a valid Kernel method.')

    def get_params(self) -> dict:
        params = dict(kernel=self._param_value(self._kernel.name),
                      alpha=self._param_value(self._alpha),
                      coefficient=self._param_value(self._coefficient),
                      degree=self._param_value(self._degree),
                      sigma=self._param_value(self._sigma))

        return params
