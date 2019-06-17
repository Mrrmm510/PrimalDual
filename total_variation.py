from typing import Tuple, List

import numpy as np


class TotalVariation:
    def __init__(self, 
            lambd: float = 1.0, 
            max_iter: int = 1000, 
            tol: float = 1e-3,
            eps: float = 1e-16,
            saturation: bool = False, 
            extended_output: bool = False):
        """
        Total Variation L1 model using the preconditioned primal dual algorithm
        
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
        self.tol = tol
        self.eps = eps
        self.saturation = saturation
        self.extended_output = extended_output
        
    def _tv(self, u: np.ndarray):
        ret = np.hstack((np.array([x - y for x, y in zip(u.T, u.T[1:])]).flatten(), 
                                 np.array([x - y for x, y in zip(u, u[1:])]).flatten()))
        return ret
        
    def _transposed_tv(self, v: np.ndarray, shape: Tuple[int, int]):
        h, w = shape[:2]
        ret = np.zeros(h * w)
        ret[:(h - 1) * w] += v[h * (w - 1):]
        ret[-(h - 1) * w:] -= v[h * (w - 1):]
        for i in range(w - 1):
            ret[i::w] += v[i*h:(i+1)*h]
            ret[i+1::w] -= v[i*h:(i+1)*h]
        return ret
    
    def _projection(self, u: np.ndarray, vmin: float, vmax: float):
        u[u < vmin] = vmin
        u[u > vmax] = vmax
        return u
    
    def _step_size(self, shape: Tuple[int, int]):
        h, w = shape[:2]
        
        tau = np.ones(h * w)
        tau[:(h - 1) * w] += 1.
        tau[-(h - 1) * w:] += 1.
        for i in range(w - 1):
            tau[i::w] += 1.
            tau[i+1::w] += 1.
        tau = 1. / tau
        sigma = np.ones(3 * h * w - h - w)
        sigma[:-h] += 1.
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
        dual = self._projection(sigma * np.hstack([self._tv(res), np.zeros(hw)]), -1, 1)
        
        # objective function
        obj = list()
        if self.extended_output:
            obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))
        
        # main loop
        for _ in range(self.max_iter):
            if self.saturation:
                u = self._projection(res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:])).reshape((h, w)), 0, 1)
            else:
                u = res - (tau * (self._transposed_tv(dual[:-hw], shape=(h, w)) + self.lambd * dual[-hw:])).reshape((h, w))
            bar_u = 2 * u - res
            dual = self._projection(dual + sigma * np.hstack([self._tv(bar_u), self.lambd * (bar_u - X).flatten()]), -1, 1)
            diff = u - res
            res = u
            if self.extended_output:
                obj.append(np.sum(np.abs(self._tv(res))) + self.lambd * np.sum(np.abs(res - X)))
            if np.linalg.norm(diff) / (np.linalg.norm(res) + self.eps) < self.tol:
                break
        return res, obj