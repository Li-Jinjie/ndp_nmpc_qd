#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: nmpc_body_rate_ctl.py
Date: 2023/4/16 9:50 AM
Description:
"""
import os
import sys
import shutil
import errno
import numpy as np
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import casadi as ca

from params import nmpc_params as CP


class NMPCBodyRateController(object):
    def __init__(self, is_build_acados=True):
        opt_model = BodyRateModel().model

        nx = opt_model.x.size()[0]
        nu = opt_model.u.size()[0]
        ny = nx + nu
        n_params = opt_model.p.size()[0]

        # get file path for acados
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        acados_models_dir = "./acados_models"
        safe_mkdir_recursive(os.path.join(os.getcwd(), acados_models_dir))
        acados_source_path = os.environ["ACADOS_SOURCE_DIR"]
        sys.path.insert(0, acados_source_path)

        # create OCP
        ocp = AcadosOcp()
        ocp.acados_include_path = acados_source_path + "/include"
        ocp.acados_lib_path = acados_source_path + "/lib"
        ocp.model = opt_model
        ocp.dims.N = CP.N_node

        # initialize parameters
        ocp.dims.np = n_params
        ocp.parameter_values = np.zeros(n_params)  # "p" is set to zeros by default

        # cost function
        # see https://docs.acados.org/python_interface/#acados_template.acados_ocp.AcadosOcpCost for details
        ocp.cost.cost_type = "EXTERNAL"  # TODO: try NONLINEAR_LS
        ocp.cost.cost_type_e = "EXTERNAL"

        # set constraints
        ocp.constraints.lbu = np.array([CP.w_min, CP.w_min, CP.w_min, CP.c_min])
        ocp.constraints.ubu = np.array([CP.w_max, CP.w_max, CP.w_max, CP.c_max])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])  # omega_x, omega_y, omega_z, collective_acceleration, fx, fy, fz
        ocp.constraints.lbx = np.array([CP.v_min, CP.v_min, CP.v_min])
        ocp.constraints.ubx = np.array([CP.v_max, CP.v_max, CP.v_max])
        ocp.constraints.idxbx = np.array([3, 4, 5])  # vx, vy, vz

        # initial state
        x_ref = np.zeros(nx)
        ocp.constraints.x0 = x_ref

        # solver options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"  # explicit Runge-Kutta integrator
        ocp.solver_options.print_level = 0
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.qp_solver_cond_N = CP.N_node
        ocp.solver_options.tf = CP.T_horizon

        # compile acados ocp
        json_file_path = os.path.join("./" + opt_model.name + "_acados_ocp.json")
        self.solver = AcadosOcpSolver(ocp, json_file=json_file_path, build=is_build_acados)

    def update(self, x0, xr, ur):
        # get x and u, set reference
        for i in range(self.solver.N):
            xr_ur = np.concatenate((xr[i, :], ur[i, :]))
            self.solver.set(i, "p", xr_ur)

        # set terminal reference of x. u will be ignored.
        u_zero = np.zeros_like(ur[0, :])
        xr_ur = np.concatenate((xr[self.solver.N, :], u_zero))
        self.solver.set(self.solver.N, "p", xr_ur)

        # self.solver.set(i, "p", fx, fy, fz)

        u0 = self.solver.solve_for_x0(x0)  # feedback, take the first action

        if self.solver.status != 0:
            raise Exception("acados acados_ocp_solver returned status {}. Exiting.".format(self.solver.status))

        return u0


class BodyRateModel(object):
    def __init__(self):
        model_name = "qd_body_rate_model"

        # model states
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        z = ca.SX.sym("z")
        vx = ca.SX.sym("vx")
        vy = ca.SX.sym("vy")
        vz = ca.SX.sym("vz")
        qw = ca.SX.sym("qw")
        qx = ca.SX.sym("qx")
        qy = ca.SX.sym("qy")
        qz = ca.SX.sym("qz")
        states = ca.vertcat(x, y, z, vx, vy, vz, qw, qx, qy, qz)
        # nx = states.size()[0]

        # reference for states
        xr = ca.SX.sym("xr")
        yr = ca.SX.sym("yr")
        zr = ca.SX.sym("zr")
        vxr = ca.SX.sym("vxr")
        vyr = ca.SX.sym("vyr")
        vzr = ca.SX.sym("vzr")
        qwr = ca.SX.sym("qwr")
        qxr = ca.SX.sym("qxr")
        qyr = ca.SX.sym("qyr")
        qzr = ca.SX.sym("qzr")
        states_r = ca.vertcat(xr, yr, zr, vxr, vyr, vzr, qwr, qxr, qyr, qzr)

        # control inputs
        wx = ca.SX.sym("wx")
        wy = ca.SX.sym("wy")
        wz = ca.SX.sym("wz")
        c = ca.SX.sym("c")
        controls = ca.vertcat(wx, wy, wz, c)
        # nu = controls.size()[0]

        # reference for inputs
        wxr = ca.SX.sym("wxr")
        wyr = ca.SX.sym("wyr")
        wzr = ca.SX.sym("wzr")
        cr = ca.SX.sym("cr")
        controls_r = ca.vertcat(wxr, wyr, wzr, cr)

        # dynamic model
        ds = ca.vertcat(
            vx,
            vy,
            vz,
            2 * (qx * qz + qw * qy) * c,
            2 * (qy * qz - qw * qx) * c,
            (1 - 2 * qx**2 - 2 * qy**2) * c - CP.gravity,
            (-wx * qx - wy * qy - wz * qz) * 0.5,
            (wx * qw + wz * qy - wy * qz) * 0.5,
            (wy * qw - wz * qx + wx * qz) * 0.5,
            (wz * qw + wy * qx - wx * qy) * 0.5,
        )

        # function
        f = ca.Function("f", [states, controls], [ds], ["state", "control_input"], ["ds"])

        # cost
        Q = np.diag([CP.Qp_xy, CP.Qp_xy, CP.Qp_z, CP.Qv_xy, CP.Qv_xy, CP.Qv_z, CP.Qq, CP.Qq, CP.Qq_z])  # dim = 9
        R = np.diag([CP.Rw, CP.Rw, CP.Rw, CP.Rc])  # dim = 3

        error_states = ca.vertcat(
            x - xr,
            y - yr,
            z - zr,
            vx - vxr,
            vy - vyr,
            vz - vzr,
            qwr * qx - qw * qxr + qyr * qz - qy * qzr,
            qwr * qy - qw * qyr - qxr * qz + qx * qzr,
            qxr * qy - qx * qyr + qwr * qz - qw * qzr,
        )
        error_controls = ca.vertcat(
            wx - wxr,
            wy - wyr,
            wz - wzr,
            c - cr,
        )

        # error_x_u = ca.vertcat(error_states, error_controls)  # TODO: compare these code with bottom
        # W = np.block([[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        # cost_f = error_x_u.T @ W @ error_x_u

        cost_f = error_states.T @ Q @ error_states + error_controls.T @ R @ error_controls
        cost_fe = error_states.T @ Q @ error_states

        # acados model
        x_dot = ca.SX.sym("x_dot", 10)
        f_impl = x_dot - f(states, controls)

        model = AcadosModel()
        model.name = model_name
        model.f_expl_expr = f(states, controls)  # CasADi expression for the explicit dynamics
        model.f_impl_expr = f_impl  # CasADi expression for the implicit dynamics
        model.x = states
        model.xdot = x_dot
        model.u = controls
        model.p = ca.vertcat(states_r, controls_r)
        model.cost_expr_ext_cost = cost_f
        model.cost_expr_ext_cost_e = cost_fe

        # constraint
        constraint = ca.types.SimpleNamespace()

        constraint.w_max = CP.w_max
        constraint.w_min = CP.w_min
        constraint.c_max = CP.c_max
        constraint.c_min = CP.c_min

        constraint.v_max = CP.v_max
        constraint.v_min = CP.v_min
        constraint.expr = ca.vcat([wx, wy, wz, c])

        self.model = model
        self.constraint = constraint


def safe_mkdir_recursive(directory, overwrite=False):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(directory):
                pass
            else:
                raise
    else:
        if overwrite:
            try:
                shutil.rmtree(directory)
            except:
                print("Error while removing directory {}".format(directory))
