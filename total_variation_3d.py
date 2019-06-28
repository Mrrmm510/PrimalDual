from typing import Tuple, List

import numpy as np
from tqdm import trange


class TotalVariation3D:
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

    def _tv(self, u: np.ndarray) -> np.ndarray:
        d, h, w = u.shape
        index1 = d * h * (w - self.length)
        index2 = index1 + d * w * (h - self.length)
        ret = np.zeros(3 * d * h * w - self.length * (h * w + w * d + d * h))
        for i, c in enumerate(self.coef):
            ret[: index1] += c * np.roll(u, -i, axis=2)[:, :, :-self.length].flatten()
            ret[index1 : index2] += c * np.roll(u, -i, axis=1)[:, :-self.length].flatten()
            ret[index2 :] += c * np.roll(u, -i, axis=0)[:-self.length].flatten()
        return ret

    def _transposed_tv(self, v: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = shape
        hw = h * w
        index1 = d * h * (w - self.length)
        index2 = index1 + d * w * (h - self.length)
        index3 = h * w * (d - self.length)
        index4 = w * (h - self.length)
        v0 = np.copy(v[:index1])
        for i in range(self.length):
            v0 = np.insert(v0, [j * (w - self.length + i) for j in range(1, h * d)], 0)
        ret = np.zeros(hw * d)
        for i, c in enumerate(self.coef):
            ret[i : len(v0) + i] += c * v0
            ret[hw * i : hw * i + index3] += c * v[index2:]
            for j in range(d):
                ret[i * w + j * hw : i * w + j * hw + index4] += c * v[index1 + j * w : index1 + j * w + index4]
        return ret

    def _step_size(self, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        d, h, w = shape
        hw = h * w
        hwd = hw * d
        index1 = w * (h - self.length)
        index2 = h * w * (d - self.length)
        abs_coef = np.abs(self.coef)
        abs_sum = np.sum(abs_coef)
        tau = np.full(hwd, self.lambd)
        t = np.zeros(w)
        for i in range(w - self.length):
            t[i : i + self.length + 1] += abs_coef
        tile = np.tile(t, h)
        for i, c in enumerate(abs_coef):
            tile[w * i : w * i + index1] += c
        tau += np.tile(tile, d)
        for i, c in enumerate(abs_coef):
            tau[hw * i : hw * i + index2] += c
        tau = 1. / tau
        sigma = np.zeros(4 * hwd - self.length * (hw + w * d + d * h))
        sigma[:-hwd] += abs_sum
        sigma[-hwd:] += self.lambd
        sigma = 1. / sigma
        return tau, sigma

    def transform(self, X: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """
        Parameters
        ----------
        X : array, shape = (d, h, w)
            a 2D image

        Returns
        ----------
        res : array, shape = (d, h, w)
            a denoised image
        obj : a list of float
            the value of the objective function of each iteration
        """
        d, h, w = X.shape[:3]
        hw = h * w
        hwd = hw * d
        tau, sigma = self._step_size((d, h, w))

        # initialize
        res = np.copy(X)
        dual = np.zeros(4 * hwd - self.length * (hw + w * d + d * h))
        dual[:-hwd] = np.clip(sigma[:-hwd] * self._tv(res), -1, 1)

        # objective function
        obj = list()
        if self.extended_output:
            obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))

        # main loop
        for _ in trange(self.max_iter):
            if self.saturation:
                u = np.clip(res - (tau * (self._transposed_tv(dual[:-hwd], shape=(d, h, w)) + self.lambd * dual[-hwd:])).reshape((d, h, w)), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[:-hwd], shape=(d, h, w)) + self.lambd * dual[-hwd:])).reshape((d, h, w))
            bar_u = 2 * u - res
            dual[:-hwd] += sigma[:-hwd] * self._tv(bar_u)
            dual[-hwd:] += sigma[-hwd:] * self.lambd * (bar_u - X).flatten()
            dual = np.clip(dual, -1, 1)
            diff = u - res
            res = u
            if self.extended_output:
                obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))
            if np.linalg.norm(diff) / (np.linalg.norm(res) + self.eps) < self.tol:
                break
        return res, obj
