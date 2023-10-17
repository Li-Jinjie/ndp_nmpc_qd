#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
@Author  : Li-Jinjie
@Time    : 2022/5/15 10:22
@File    : polym_optimizer
@Email   : lijinjie362@outlook.com
@Description:  None
"""
import numpy as np
from enum import Enum, unique


@unique
class MinMethod(Enum):
    SNAP = "snap"
    JERK = "jerk"
    ACCEL = "acceleration"
    VEL = "velocity"


class PolymOptimizer:
    def __init__(self, method: MinMethod) -> None:

        if method == MinMethod.SNAP:
            self.ord_deriv = 4  # order of derivative, Nd
        elif method == MinMethod.JERK:
            self.ord_deriv = 3
        elif method == MinMethod.ACCEL:
            self.ord_deriv = 2
        elif method == MinMethod.VEL:
            self.ord_deriv = 1
        else:
            print("[ERROR] Non-existed trajectory generation method!")

        self.ord_polym = 2 * self.ord_deriv - 1  # N
        self.num_wpt = int()  # M +1

    def get_coeff(self, wpt_seq: np.array):
        """

        Args:
            wpt_seq: 1-D nd.array
        """
        self.num_wpt = len(wpt_seq) - 1

        # alpha = [10,11,12,13,14,15,16,17,   21,22,23,24,25,26,27,28,   31...]
        num_params = self.num_wpt * (self.ord_polym + 1)
        a_mat = np.zeros([num_params, num_params])
        b_vec = np.zeros([num_params, 1])

        m = self.num_wpt
        n = self.ord_polym

        # constraint: pi(Si-1)=wi-1
        t = 0
        k = 0
        row = 0
        for i in range(m):
            col_shift = i * (n + 1)
            a_mat[row, col_shift : col_shift + (n + 1)] = self.get_poly_params(k, t)
            b_vec[row, 0] = wpt_seq[i]
            row += 1

        # constraint: pi(Si)=wi
        t = 1
        k = 0
        for i in range(m):
            col_shift = i * (n + 1)
            a_mat[row, col_shift : col_shift + (n + 1)] = self.get_poly_params(k, t)
            b_vec[row, 0] = wpt_seq[i + 1]
            row += 1

        # constraint: p1^k(S_0)=0  t=0?
        t = 0
        for k in range(1, self.ord_deriv):
            col_shift = 0  # only the first segment
            a_mat[row, col_shift : col_shift + (n + 1)] = self.get_poly_params(k, t)
            b_vec[row, 0] = 0
            row += 1

        # constraint: pM^k(S_M)=0  t=1
        t = 1
        for k in range(1, self.ord_deriv):
            col_shift = (n + 1) * (m - 1)  # only the last segment
            a_mat[row, col_shift : col_shift + (n + 1)] = self.get_poly_params(k, t)
            b_vec[row, 0] = 0
            row += 1

        # constraint: pi^k(S_i) - p(i+1)^k(S_i) = 0

        for i in range(m - 1):
            col_shift = i * (n + 1)
            for k in range(1, n):
                t = 1  # the final point of the last segment
                a_mat[row, col_shift : col_shift + (n + 1)] = self.get_poly_params(k, t)
                t = 0  # the first point of the now segment
                a_mat[row, col_shift + (n + 1) : col_shift + 2 * (n + 1)] = -self.get_poly_params(k, t)
                b_vec[row, 0] = 0
                row += 1

        return np.linalg.inv(a_mat) @ b_vec

    def get_poly_params(self, deriv_num: int, t: float) -> np.array:
        """obtain params of polynomial
        c0 + t*c1 + t^2*c2 + t^3*c3 + ... + t^N*cN
        refer to https://www.coursera.org/learn/robotics-flight/discussions/forums/nv8Y6Cj3EeaZ8Apto8QB_w/threads/hcXrPs9yEeWwoQrbIHhKaQ

        Args:
            deriv_num: the requested derivative number
            t: the actual value of normalized time (this can be anything, not just 0 or 1).

        Returns:
            params: params of polynomial

        """
        n = self.ord_polym + 1
        params = np.zeros([n, 1])
        orders = np.zeros([n, 1])

        # init
        for i in range(1, n + 1):
            orders[i - 1] = i - 1
            params[i - 1] = 1

        # derivative
        for j in range(1, deriv_num + 1):
            for i in range(1, n + 1):
                params[i - 1] *= orders[i - 1]
                if orders[i - 1] > 0:
                    orders[i - 1] -= 1

        # put t value
        for i in range(1, n + 1):
            params[i - 1] *= np.power(t, orders[i - 1])

        params = params.T

        return params
