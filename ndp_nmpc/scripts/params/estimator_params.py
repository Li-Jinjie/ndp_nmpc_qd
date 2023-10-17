#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: estimator_params.py
Date: 2023/4/18 11:04 AM
Description:
"""
import numpy as np
from .fhnp_params import gravity, mass

# k_throttle_init = mass * gravity / 0.35  --> in real-world experiment
k_throttle_init = 50.0  # more suitable for simulation

ts_est = 0.02  # 50Hz

R = 1.225  # 0.1225 (m/s^2)^2  come from EKF2_ACC_NOISE
Q = np.diag([0.1, 0.1])
