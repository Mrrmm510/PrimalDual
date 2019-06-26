from typing import Tuple, List

import numpy as np


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

    def _tv(self, u: np.ndarray) -> np.ndarray:
        h, w = u.shape[:2]
        length = len(self.coef) - 1
        ret = np.zeros(2 * h * w - length * (h + w))
        for i, c in enumerate(self.coef):
            ret[: h * w - length * h] += c * np.roll(u, -i, axis=1)[:, :-1 * length].flatten()
            ret[h * w - length * h :] += c * np.roll(u, -i, axis=0)[:-1 * length].flatten()
        return ret

    def _transposed_tv(self, v: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape[:2]
        hw = h * w
        length = len(self.coef) - 1
        u = h * (w - length)
        l = w * (h - length)
        ret = np.zeros(hw)
        v0 = np.copy(v[:u])
        for i in range(length):
            v0 = np.insert(v0, [j * (w - length + i) for j in range(1, h)], 0)
        for i, c in enumerate(self.coef):
            ret[i : len(v0) + i] += c * v0
            ret[w * i : l + w * i] += c * v[u:]
        return ret

    def _step_size(self, shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        h, w = shape[:2]
        hw = h * w
        abs_coef = np.abs(self.coef)
        abs_sum = np.sum(abs_coef)
        tau = np.full(h * w, self.lambd)
        tile = np.zeros(w)
        length = len(self.coef) - 1
        l = w * (h - length)
        for i in range(w - length):
            tile[i : i + length + 1] += abs_coef
        tau += np.tile(tile, h)
        for i, c in enumerate(abs_coef):
            tau[w * i : l + w * i] += c
        tau = 1. / tau
        sigma = np.zeros(3 * h * w - length * (h + w))
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
        res = np.copy(X)
        dual = np.zeros(3 * h * w - (len(self.coef) - 1) * (h + w))
        dual[:-hw] = np.clip(sigma[:-hw] * self._tv(res), -1, 1)

        # objective function
        obj = list()
        if self.extended_output:
            obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))

        # main loop
        for _ in range(self.max_iter):
            if self.saturation:
                u = np.clip(res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:])).reshape((h, w)), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:])).reshape((h, w))
            bar_u = 2 * u - res
            dual[:-hw] += sigma[:-hw] * self._tv(bar_u)
            dual[-hw:] += sigma[-hw:] * self.lambd * (bar_u - X).flatten()
            dual = np.clip(dual, -1, 1)
            diff = u - res
            res = u
            if self.extended_output:
                obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))
            if np.linalg.norm(diff) / (np.linalg.norm(res) + self.eps) < self.tol:
                break
        return res, obj
