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


class HoverThrottleEstimator:
    def __init__(self, ts: float, mass: float, k_thrust_init: float) -> None:
        # differentiator to get a_z
        self.vz_diff = Differentiator(ts)

        # init guess
        self.x = np.array([[0], [k_thrust_init]])  # [[T], [k_param]]

        # matrices
        self.P_mtx = np.eye(2)
        self.K = None

        self.Phi_mtx = np.zeros([2, 2])
        self.Phi_mtx[1, 1] = 1
        self.G = 0
        self.Gamma_mtx = np.eye(2)
        self.H_mtx = np.array([[1 / mass, 0]])

        # parameters
        self.R = 1.225  # 0.1225 (m/s^2)^2  come from EKF2_ACC_NOISE
        self.Q = np.diag([0.1, 0.1])

    def update(self, acc_sp_z: float, throttle: float) -> tuple:
        if 0.1 < throttle < 1:
            z = acc_sp_z
            self.Phi_mtx[0, 1] = throttle

            # Kalman Filter
            self.P_mtx = self.Phi_mtx @ self.P_mtx @ self.Phi_mtx.T + self.Gamma_mtx @ self.Q @ self.Gamma_mtx.T
            self.K = self.P_mtx @ self.H_mtx.T @ np.linalg.inv(self.H_mtx @ self.P_mtx @ self.H_mtx.T + self.R)
            self.x = self.Phi_mtx @ self.x
            self.x = self.x + self.K @ (z - self.H_mtx @ self.x)
            self.P_mtx = (np.eye(2) - self.K @ self.H_mtx) @ self.P_mtx

        k_param = self.x.item(1)

        return k_param, self.x, self.P_mtx
