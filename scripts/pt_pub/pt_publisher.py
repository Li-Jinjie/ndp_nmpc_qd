#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: pt_publisher.py
Date: 2023/4/18 2:28 PM
Description:
"""
import numpy as np
import rospy
import tf_conversions

from oop_qd_onbd.msg import TrajPt, TrajFullStatePt
from ..params import fhnp_params as AP, nmpc_params as CP
from .base_pt_publisher import BasePtPublisher
from .polym_optimizer import MinMethod


class FullStatePtPublisher(BasePtPublisher):
    def __init__(self, xyz_method=MinMethod.SNAP, yaw_method=MinMethod.ACCEL):
        super().__init__(xyz_method, yaw_method)

    def get_full_state_pt(self, ros_t: rospy.Time, is_pred: bool = False) -> TrajFullStatePt:
        traj_pt = self.get_pt(ros_t, is_pred)
        traj_full_state_pt = diff_flatness(traj_pt)

        return traj_full_state_pt


class NMPCRefPublisher(FullStatePtPublisher):
    def __init__(self, xyz_method=MinMethod.SNAP, yaw_method=MinMethod.ACCEL):
        super().__init__(xyz_method, yaw_method)

    def get_nmpc_pts(self, ros_t: rospy.Time) -> (np.Array, np.Array):

        xr = np.zeros([CP.N_node + 1, CP.n_states])
        ur = np.zeros([CP.N_node, CP.n_controls])

        for i in range(CP.N_node + 1):

            is_pred = False if i == 0 else True  # the first state may change is_activate flag, the others are not.
            traj_full_pt: TrajFullStatePt = self.get_full_state_pt(ros_t, is_pred=is_pred)
            xr[i, :] = np.array(
                [
                    traj_full_pt.pose.position.x,
                    traj_full_pt.pose.position.y,
                    traj_full_pt.pose.position.z,
                    traj_full_pt.twist.linear.x,
                    traj_full_pt.twist.linear.y,
                    traj_full_pt.twist.linear.z,
                    traj_full_pt.pose.orientation.w,
                    traj_full_pt.pose.orientation.x,
                    traj_full_pt.pose.orientation.y,
                    traj_full_pt.pose.orientation.z,
                ]
            )

            # last stage has no desired u
            if i < CP.N_node:
                ur[i, :] = np.array(
                    [
                        traj_full_pt.twist.angular.x,
                        traj_full_pt.twist.angular.y,
                        traj_full_pt.twist.angular.z,
                        traj_full_pt.collective_force / AP.mass,
                    ]
                )

        return xr, ur


def diff_flatness(traj_pt: TrajPt) -> TrajFullStatePt:
    """generate full states from [pos, vel, acc, jerk], ENU coordinates

    Args:
        traj_pt:

    Returns:
        traj_full_state [pos, vel, attitude_q, body_rate, collective_force]
    """

    t_des = np.array([[traj_pt.accel.x + 0.0], [traj_pt.accel.y + 0.0], [traj_pt.accel.z + AP.gravity]])
    t_des_norm = np.linalg.norm(t_des, ord=2)

    if t_des_norm == 0:
        raise ValueError("t_des is empty!")

    z_b = t_des / t_des_norm

    u1 = t_des_norm * AP.mass

    x_c = np.array([[np.cos(traj_pt.yaw), np.sin(traj_pt.yaw), 0]]).T
    zx_cross = np.cross(z_b, x_c, axis=0)
    norm_zx_cross = np.linalg.norm(zx_cross, ord=2)

    if norm_zx_cross == 0:
        raise ValueError("norm of zx_cross is empty!")

    y_b = zx_cross / norm_zx_cross

    x_b = np.cross(y_b, z_b, axis=0)
    R_wb = np.concatenate((x_b, y_b, z_b), axis=1)

    np_jerk = np.array([[traj_pt.jerk.x], [traj_pt.jerk.y], [traj_pt.jerk.z]])

    h_omega = (AP.mass / u1) * (np_jerk - np.dot(z_b.T, np_jerk) * z_b)
    p = -np.dot(h_omega.T, y_b)
    q = np.dot(h_omega.T, x_b)
    r = traj_pt.yaw_dot * np.dot(np.array([[0, 0, 1]]), z_b)

    traj_full_state = TrajFullStatePt()
    traj_full_state.pose.position = traj_pt.position
    traj_full_state.twist.linear = traj_pt.velocity

    t_mtx = np.zeros([4, 4])
    t_mtx[0:3, 0:3] = R_wb
    t_mtx[3, 3] = 1
    q_xyzw = tf_conversions.transformations.quaternion_from_matrix(t_mtx)

    # change signs to match px4
    traj_full_state.pose.orientation.x = -q_xyzw[0]
    traj_full_state.pose.orientation.y = -q_xyzw[1]
    traj_full_state.pose.orientation.z = -q_xyzw[2]
    traj_full_state.pose.orientation.w = -q_xyzw[3]

    traj_full_state.twist.angular.x = p.item()
    traj_full_state.twist.angular.y = q.item()
    traj_full_state.twist.angular.z = r.item()

    traj_full_state.collective_force = u1

    return traj_full_state
