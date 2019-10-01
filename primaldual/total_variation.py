from typing import Tuple

import numpy as np
from tqdm import trange


class TotalVariation:
    """
    Total Variation L1 model using the preconditioned primal dual algorithm
    """
    def __init__(self,
            lambd: float = 1.0,
            max_iter: int = 1000,
            coef: np.ndarray = np.array([1, -1]),
            saturation: bool = False,
            extended_output: bool = False):
        """
        Parameters
        ----------
        lambd : float
            A regularization parameter.
        max_iter : int
            The maximum number of iterations.
        coef : np.ndarray
            [1, -1] for total valiation regularization
        saturation : bool
            If True, output will be in a range [0, 1].
        extended_output : bool
            If True, return the value of the objective function of each iteration.
        """
        self.lambd = lambd
        self.max_iter = max_iter
        self.coef = coef
        self.saturation = saturation
        self.extended_output = extended_output

        self.length = len(coef) - 1
        # objective function value
        self.obj = list()

    def _tv(self, u: np.ndarray) -> np.ndarray:
        h, w = u.shape
        ret = np.zeros((2 * h, w))
        for i, c in enumerate(self.coef):
            ret[: h, : w - self.length] += c * u[:, i : w - self.length + i]
            ret[h : 2 * h - self.length] += c * u[i : h - self.length + i]
        return ret

    def _transposed_tv(self, v: np.ndarray) -> np.ndarray:
        h2, w = v.shape
        h = h2 // 2
        ret = np.zeros((h, w))
        for i, c in enumerate(self.coef):
            ret[:, i : w - self.length + i] += c * v[: h, : w - self.length]
            ret[i : h - self.length + i] += c * v[h : 2 * h - self.length]
        return ret

    def _step_size(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape
        abs_coef = np.abs(self.coef)
        abs_sum = np.sum(abs_coef)

        tau = np.full((h, w), self.lambd + 2 * abs_sum)
        for i in range(self.length):
            tau[i] -= np.sum(abs_coef[i + 1 :])
            tau[-(i + 1)] -= np.sum(abs_coef[: -(i + 1)])
            tau[:, i] -= np.sum(abs_coef[i + 1 :])
            tau[:, -(i + 1)] -= np.sum(abs_coef[: -(i + 1)])
        tau = 1. / tau

        sigma = np.zeros((3 * h, w))
        sigma[:2 * h] += abs_sum
        sigma[2 * h:] += self.lambd
        sigma = 1. / sigma
        return tau, sigma

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : array, shape = (h, w)
            a 2D image

        Returns
        ----------
        res : array, shape = (h, w)
            a denoised image
        """
        h, w = X.shape
        h2 = h * 2
        tau, sigma = self._step_size((h, w))

        # initialize
        res = np.copy(X)
        dual = np.zeros((3 * h, w))
        dual[: h2] = np.clip(sigma[: h2] * self._tv(res), -1, 1)

        # store objective function value if necessary
        if self.extended_output:
            self.obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))

        # main loop
        for _ in trange(self.max_iter):
            if self.saturation:
                u = np.clip(res - (tau * (self._transposed_tv(dual[: h2]) + self.lambd * dual[h2 :])), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[: h2]) + self.lambd * dual[h2 :]))
            bar_u = 2 * u - res
            dual[: h2] += sigma[: h2] * self._tv(bar_u)
            dual[h2 :] += sigma[h2 :] * self.lambd * (bar_u - X)
            dual = np.clip(dual, -1, 1)
            res = u

            # store objective function value if necessary
            if self.extended_output:
                self.obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))
        return res
