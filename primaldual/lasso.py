from typing import Tuple, List

import numpy as np
from tqdm import trange


class Lasso:
    """
    Lasso regression model using the preconditioned primal dual algorithm
    """
    def __init__(self,
            alpha: float = 1.0,
            beta: float = 0.5,
            max_iter: int = 1000,
            extended_output: bool = False):
        """
        Parameters
        ----------
        alpha : float
            A regularization parameter.
        beta : float in [0.0, 2.0]
            A parameter used in step size.
        max_iter : int
            The maximum number of iterations.
        extended_output : bool
            If True, return the value of the objective function of each iteration.
        """
        np.random.seed(0)
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.extended_output = extended_output

        self.coef_ = None
        self.objective_function = list()

    def _step_size(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, n_features = X.shape
        abs_X = np.abs(X)

        tau = np.sum(abs_X ** self.beta, axis=0)
        tau += self.alpha ** self.beta
        tau = 1 / tau

        sigma = np.empty(n_samples + n_features, dtype=X.dtype)
        sigma[:n_samples] = np.sum(abs_X ** (2 - self.beta), axis=1)
        sigma[n_samples:] = self.alpha ** (2 - self.beta)
        sigma = 1. / sigma

        return tau, sigma

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
        y : array, shape = (n_samples, )
        """
        n_samples, n_features = X.shape
        n_samples_inv = 1 / n_samples
        bar_X = X * n_samples_inv
        bar_y = y * n_samples_inv
        tau, sigma = self._step_size(bar_X)

        # initialize
        res = np.zeros(n_features, dtype=X.dtype)
        dual = np.zeros(n_samples + n_features, dtype=X.dtype)
        dual[:n_samples] = np.clip(-sigma[:n_samples] * y * n_samples_inv, -1, 1)

        # objective function
        if self.extended_output:
            self.objective_function.append(
                np.sum(np.abs(bar_y)) + self.alpha * np.sum(np.abs(res))
            )

        # main loop
        for _ in trange(self.max_iter):
            w = res - tau * (bar_X.T.dot(dual[:n_samples]) + self.alpha * dual[n_samples:])
            bar_w = 2 * w - res
            dual[:n_samples] = dual[:n_samples] + sigma[:n_samples] * (bar_X.dot(bar_w) - bar_y)
            dual[n_samples:] = dual[n_samples:] + sigma[n_samples:] * bar_w * self.alpha
            dual = np.clip(dual, -1, 1)
            res = w
            if self.extended_output:
                self.objective_function.append(
                    np.sum(np.abs(bar_X.dot(w) - bar_y)) + self.alpha * np.sum(np.abs(res))
                )
        self.coef_ = res
