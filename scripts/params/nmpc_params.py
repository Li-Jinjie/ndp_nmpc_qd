#!/usr/bin/env python
# -*- encoding: ascii -*-
from . import fhnp_params as QD

gravity = QD.gravity
mass = QD.mass

# basic params
N_node = 20
T_horizon = 2
ts_nmpc = 0.01  # 100 Hz
th_pred = T_horizon / N_node  # seconds

n_states = 10
n_controls = 4

# params for constraints
# constraint
w_max = 1  # 1 rad/s ~ 57 deg/s
w_min = -1
c_max = QD.c_max
c_min = 0

v_max = 2  # TODO: make greater
v_min = -2

# params for the cost function
Qp_xy = 300  # 0
Qp_z = 400  # 500
Qv_xy = 10  # 0
Qv_z = 10  # 100
Qq = 10  # 400
Qq_z = 10
Rw = 10  # 10
Rc = 5
