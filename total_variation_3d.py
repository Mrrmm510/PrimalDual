from typing import Tuple, List

import numpy as np
from tqdm import trange


class TotalVariation3D:
    """
    Total Variation L1 model using the preconditioned primal dual algorithm
    """
    def __init__(self,
            lambd: float = 1.0,
            param_d: float = 1.0,
            param_h: float = 1.0,
            param_w: float = 1.0,
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
        self.param_d = param_d
        self.param_h = param_h
        self.param_w = param_w
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

    def _tv(self, u: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = shape
        hw = h * w
        dhw = d * hw
        index1 = (dhw - self.length) * (self.param_w > 0)
        index2 = w * (h - self.length) * (self.param_h > 0)
        shape = index1 + d * index2 + (hw * (d - self.length)) * (self.param_d > 0)
        ret = np.empty(shape)

        if self.param_w > 0:
            ret[: index1] = self._tv_one(u, step=1) * self.param_w
            for i in range(self.length):
                ret[: index1][w - self.length + i::w] = 0

        if self.param_h > 0:
            for i in range(d):
                ret[index1 + i * index2: index1 + (i + 1) * index2] = self._tv_one(u[i * hw : (i + 1) * hw], step=w) * self.param_h

        if self.param_d > 0:
            ret[index1 + d * index2 :] = self._tv_one(u, step=hw) * self.param_d
        return ret

    def _transposed_tv_one(self, v: np.ndarray, step: int) -> np.ndarray:
        length = len(v)
        ret = np.zeros(length + self.length * step)
        for i, c in enumerate(self.coef):
            ret[i * step: i * step + length] += c * v
        return ret

    def _transposed_tv(self, v: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
        d, h, w = shape
        hw = h * w
        dhw = d * hw
        index1 = (dhw - self.length) * (self.param_w > 0)
        index2 = w * (h - self.length) * (self.param_h > 0)
        ret = np.zeros(dhw)

        if self.param_w > 0:
            ret += self._transposed_tv_one(v[: index1], step=1) * self.param_w

        if self.param_h > 0:
            for i in range(d):
                ret[i * hw: (i + 1) * hw] += self._transposed_tv_one(v[index1 + i * index2 : index1 + (i + 1) * index2], step=w) * self.param_h

        if self.param_d > 0:
            ret += self._transposed_tv_one(v[index1 + d * index2 :], step=hw) * self.param_d
        return ret

    def _step_size(self, shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        d, h, w = shape
        hw = h * w
        dhw = d * hw
        index2 = w * (h - self.length) * (self.param_h > 0)
        index3 = h * w * (d - self.length)
        shape = (dhw - self.length) * (self.param_w > 0) + d * index2 + (hw * (d - self.length)) * (self.param_d > 0)
        abs_coef = np.abs(self.coef)
        abs_sum = np.sum(abs_coef)

        tau = np.full(dhw, self.lambd)

        if self.param_w > 0:
            tau += abs_sum
            for i in range(self.length):
                tau[i] -= np.sum(abs_coef[i + 1 :])
                tau[-(i + 1)] -= np.sum(abs_coef[: -(i + 1)])

        if self.param_h > 0:
            tile = np.zeros(hw)
            for i, c in enumerate(abs_coef):
                tile[w * i : w * i + index2] += c
            tau += np.tile(tile, d)

        if self.param_d > 0:
            for i, c in enumerate(abs_coef):
                tau[hw * i : hw * i + index3] += c
        tau = 1. / tau

        sigma = np.zeros(dhw + shape)
        sigma[:-dhw] += abs_sum
        sigma[-dhw:] += self.lambd
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
        dhw = hw * d
        shape = (dhw - self.length) * (self.param_w > 0) \
            + d * w * (h - self.length) * (self.param_h > 0) \
            + (hw * (d - self.length)) * (self.param_d > 0)
        tau, sigma = self._step_size((d, h, w))

        # initialize
        x = X.flatten()
        res = np.copy(x)
        dual = np.zeros(dhw + shape)
        dual[:-dhw] = np.clip(sigma[:-dhw] * self._tv(res, (d, h, w)), -1, 1)

        # objective function
        obj = list()
        if self.extended_output:
            obj.append(np.sum(np.abs(self._tv(res, (d, h, w)))) + self.lambd * np.sum(np.abs(res - x)))

        # main loop
        for _ in trange(self.max_iter):
            if self.saturation:
                u = np.clip(res - (tau * (self._transposed_tv(dual[:-dhw], shape=(d, h, w)) + self.lambd * dual[-dhw:])), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[:-dhw], shape=(d, h, w)) + self.lambd * dual[-dhw:]))
            bar_u = 2 * u - res
            dual[:-dhw] += sigma[:-dhw] * self._tv(bar_u, (d, h, w))
            dual[-dhw:] += sigma[-dhw:] * self.lambd * (bar_u - x)
            dual = np.clip(dual, -1, 1)
            diff = u - res
            res = u
            if self.extended_output:
                obj.append(np.sum(np.abs(self._tv(res, (d, h, w)))) + self.lambd * np.sum(np.abs(res - x)))
            # if np.linalg.norm(diff) / (np.linalg.norm(res) + self.eps) < self.tol:
            #     break
        return res.reshape((d, h, w)), obj
