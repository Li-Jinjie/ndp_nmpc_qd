#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: hover_throttle_estimator.py
Date: 2023/4/18 10:35 AM
Description:
"""

import numpy as np
from .differentiator import Differentiator
from ..params import estimator_params as EP


class HoverThrottleEstimator:
    def __init__(self, ts: float) -> None:
        # differentiator to get a_z
        self.vz_diff = Differentiator(ts)

        # init guess
        self.x = np.array([[0], [EP.k_throttle_init]])  # [[f_collect], [k_throttle]]
        self.P_mtx = np.eye(2)

        self.K = None

        # define model
        self.Phi_mtx = np.zeros([2, 2])
        self.Phi_mtx[1, 1] = 1
        self.G = 0
        self.Gamma_mtx = np.eye(2)
        self.H_mtx = np.array([[1 / EP.mass, 0]])

        # parameters
        self.R = EP.R
        self.Q = EP.Q

    def update(self, vz: float, throttle: float) -> tuple:
        az = self.vz_diff.update(vz)

        if 0.1 < throttle < 1:
            z = az + EP.gravity
            self.Phi_mtx[0, 1] = throttle

            # Kalman Filter
            self.P_mtx = self.Phi_mtx @ self.P_mtx @ self.Phi_mtx.T + self.Gamma_mtx @ self.Q @ self.Gamma_mtx.T
            self.K = self.P_mtx @ self.H_mtx.T @ np.linalg.inv(self.H_mtx @ self.P_mtx @ self.H_mtx.T + self.R)
            self.x = self.Phi_mtx @ self.x
            self.x = self.x + self.K @ (z - self.H_mtx @ self.x)
            self.P_mtx = (np.eye(2) - self.K @ self.H_mtx) @ self.P_mtx

        k_throttle = self.x.item(1)

        return k_throttle, self.x, self.P_mtx
