from typing import Tuple, List

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
            tol: float = 1e-3,
            eps: float = 1e-16,
            saturation: bool = False,
            extended_output: bool = False):
        """
        Parameters
        ----------
        lambd : float
            A regularization parameter.
        max_iter : int
            The maximum number of iterations.
        tol : float
            Tolerance for stopping criterion.
        eps : float
            a value to avoid divided by zero error.
        saturation : bool
            If True, output will be in a range [0, 1].
        extended_output : bool
            If True, return the value of the objective function of each iteration.
        """
        self.lambd = lambd
        self.max_iter = max_iter
        self.coef = coef
        self.tol = tol
        self.eps = eps
        self.saturation = saturation
        self.extended_output = extended_output

        self.length = len(coef) - 1

    def _tv_one(self, u: np.ndarray, step: int) -> np.ndarray:
        length = len(u) - self.length * step
        ret = self.coef[0] * u[: length]
        for i, c in enumerate(self.coef[1:]):
            ret += c * u[(i + 1) * step: (i + 1) * step + length]
        return ret

    def _tv(self, u: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        hw = h * w
        ret = np.empty(2 * hw - self.length * (w + 1))
        ret[: hw - self.length] = self._tv_one(u, step=1)
        for i in range(self.length):
            ret[: hw - self.length][w - self.length + i::w] = 0
        ret[hw - self.length :] = self._tv_one(u, step=w)
        return ret

    def _transposed_tv_one(self, v: np.ndarray, step: int) -> np.ndarray:
        length = len(v)
        ret = np.zeros(length + self.length * step)
        for i, c in enumerate(self.coef):
            ret[i * step: i * step + length] += c * v
        return ret

    def _transposed_tv(self, v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        index = h * w - self.length
        ret = self._transposed_tv_one(v[index :], step=w)
        ret += self._transposed_tv_one(v[: index], step=1)
        return ret

    def _step_size(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape
        hw = h * w
        index = w * (h - self.length)
        abs_coef = np.abs(self.coef)
        abs_sum = np.sum(abs_coef)

        tau = np.full(hw, self.lambd + abs_sum)
        for i in range(self.length):
            tau[i] -= np.sum(abs_coef[i + 1 :])
            tau[-(i + 1)] -= np.sum(abs_coef[: -(i + 1)])
        for i, c in enumerate(abs_coef):
            tau[w * i : w * i + index] += c
        tau = 1. / tau

        sigma = np.zeros(3 * hw - self.length * (w + 1))
        sigma[:-hw] += abs_sum
        sigma[-hw:] += self.lambd
        sigma = 1. / sigma
        return tau, sigma

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Parameters
        ----------
        X : array, shape = (h, w)
            a 2D image

        Returns
        ----------
        res : array, shape = (h, w)
            a denoised image
        obj : a list of float
            the value of the objective function of each iteration
        """
        h, w = X.shape[:2]
        hw = h * w
        tau, sigma = self._step_size((h, w))

        # initialize
        x = X.flatten()
        res = np.copy(x)
        dual = np.zeros(3 * hw - self.length * (w + 1))
        dual[:-hw] = np.clip(sigma[:-hw] * self._tv(res, (h, w)), -1, 1)

        # objective function
        obj = list()
        if self.extended_output:
            obj.append(np.sum(np.abs(self._tv(res, (h, w)))) + self.lambd * np.sum(np.abs(res - x)))

        # main loop
        for _ in trange(self.max_iter):
            if self.saturation:
                u = np.clip(res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:])), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:]))
            bar_u = 2 * u - res
            dual[:-hw] += sigma[:-hw] * self._tv(bar_u, (h, w))
            dual[-hw:] += sigma[-hw:] * self.lambd * (bar_u - x)
            dual = np.clip(dual, -1, 1)
            diff = u - res
            res = u
            if self.extended_output:
                obj.append(np.sum(np.abs(self._tv(res, (h, w)))) + self.lambd * np.sum(np.abs(res - x)))
            if np.linalg.norm(diff) / (np.linalg.norm(res) + self.eps) < self.tol:
                break
        return res.reshape(h, w), obj
