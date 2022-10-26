import numpy as np
from abc import abstractmethod, ABC
from typing import Union, Tuple


class NotFittedException(Exception):
    pass


class InvalidNumOfComponentsException(Exception):
    pass


class OneSamplePassedException(Exception):
    pass


class _Model(object):
    def __init__(self):
        pass

    @staticmethod
    def _param_value(param: any) -> Union[any, str]:
        return param if param is not None else 'auto'

    def get_params(self) -> dict:
        return {}


class _Decomposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def _pov_to_n_components(self) -> int:
        pass

    @abstractmethod
    def _check_n_components(self) -> None:
        pass

    @abstractmethod
    def _clean_eigs(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def transform(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        pass