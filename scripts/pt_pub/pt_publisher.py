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
from typing import Tuple
from nav_msgs.msg import Odometry

from oop_qd_onbd.msg import TrajPt, TrajFullStatePt, TrajCoefficients
from params import fhnp_params as AP, nmpc_params as CP
from .base_pt_publisher import BasePtPublisher
from .polym_optimizer import MinMethod


class FullStatePtPublisher(BasePtPublisher):
    def __init__(self, xyz_method=MinMethod.SNAP, yaw_method=MinMethod.ACCEL):
        super().__init__(xyz_method, yaw_method)

    def get_traj_full_state_pt(
        self, ros_t: rospy.Time, t_pred: float = 0.0, traj_pt_now: TrajPt = None
    ) -> TrajFullStatePt:
        traj_pt = self.get_traj_pt(ros_t, t_pred, traj_pt_now)
        traj_full_state_pt = diff_flatness(traj_pt)

        return traj_full_state_pt


class NMPCRefPublisher(FullStatePtPublisher):
    def __init__(self, xyz_method=MinMethod.SNAP, yaw_method=MinMethod.ACCEL):
        super().__init__(xyz_method, yaw_method)
        self.x_long_list = []
        self.u_long_list = []

    def gen_fix_pt_ref(self, odom: Odometry):
        """first ref, all equal to current odom

        :param odom:
        :return:
        """
        self.x_long_list.clear()
        self.u_long_list.clear()

        x_1 = self.odom_2_nmpc_x(odom)
        u_1 = np.array([0, 0, 0, CP.mass * CP.gravity])
        self.x_long_list = [x_1] * CP.long_list_size
        self.u_long_list = [u_1] * CP.long_list_size

        x_r, u_r = self._get_nmpc_ref_from_long_list()
        return x_r, u_r

    def reset(self, traj_coeff: TrajCoefficients, ros_t: rospy.Time) -> None:
        super().reset(traj_coeff, ros_t)
        self._gen_long_list_w_traj()
        super().reset(traj_coeff, rospy.Time.now())

    def _gen_long_list_w_traj(self):
        self.x_long_list.clear()
        self.u_long_list.clear()

        for i in range(CP.long_list_size - 1):
            ros_t_pred = self.start_ros_t + rospy.Duration.from_sec(i * CP.ts_nmpc)
            traj_full_pt: TrajFullStatePt = self.get_traj_full_state_pt(ros_t_pred)
            x, u = self.traj_full_pt_2_x_u(traj_full_pt)
            self.x_long_list.append(x)
            self.u_long_list.append(u)

        self.x_long_list.insert(0, self.x_long_list[0])
        self.u_long_list.insert(0, self.u_long_list[0])  # duplicate the first one

        self.is_activated = False  # must be reset() again to reset the ros_t

    def get_nmpc_pts(self, ros_t: rospy.Time) -> (np.ndarray, np.ndarray):
        # remove the first element
        self.x_long_list.pop(0)
        self.u_long_list.pop(0)

        # get new point
        ros_t_pred = ros_t + rospy.Duration.from_sec(CP.T_horizon)
        traj_pt_now = self.x_2_traj_pt(self.x_long_list[0])
        traj_full_pt: TrajFullStatePt = self.get_traj_full_state_pt(
            ros_t_pred, t_pred=CP.T_horizon, traj_pt_now=traj_pt_now
        )
        x, u = self.traj_full_pt_2_x_u(traj_full_pt)

        # add new element
        self.x_long_list.append(x)
        self.u_long_list.append(u)

        xr, ur = self._get_nmpc_ref_from_long_list()

        return xr, ur

    def _get_nmpc_ref_from_long_list(self) -> Tuple[np.ndarray, np.ndarray]:
        xr_list = self.x_long_list[CP.xr_list_index]
        ur_list = self.u_long_list[CP.xr_list_index]
        ur_list.pop(-1)
        return np.array(xr_list), np.array(ur_list)

    @staticmethod
    def odom_2_nmpc_x(odom: Odometry) -> np.ndarray:
        x = np.array(
            [
                odom.pose.pose.position.x,
                odom.pose.pose.position.y,
                odom.pose.pose.position.z,
                odom.twist.twist.linear.x,
                odom.twist.twist.linear.y,
                odom.twist.twist.linear.z,
                -odom.pose.pose.orientation.w,
                -odom.pose.pose.orientation.x,
                -odom.pose.pose.orientation.y,
                -odom.pose.pose.orientation.z,
            ]
        )  # Quaternion: from px4 convention to ros convention

        return x

    @staticmethod
    def traj_full_pt_2_x_u(traj_full_pt: TrajFullStatePt) -> (np.ndarray, np.ndarray):
        x = np.array(
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
        u = np.array(
            [
                traj_full_pt.twist.angular.x,
                traj_full_pt.twist.angular.y,
                traj_full_pt.twist.angular.z,
                traj_full_pt.collective_force / AP.mass,
            ]
        )
        return x, u

    @staticmethod
    def x_2_traj_pt(x: np.ndarray) -> TrajPt:
        traj_pt = TrajPt()
        traj_pt.position.x = x[0]
        traj_pt.position.y = x[1]
        traj_pt.position.z = x[2]
        traj_pt.velocity.x = x[3]
        traj_pt.velocity.y = x[4]
        traj_pt.velocity.z = x[5]
        return traj_pt

    @staticmethod
    def x_2_full_state_pt(x: np.ndarray) -> TrajFullStatePt:
        traj_full_state_pt = TrajFullStatePt()
        traj_full_state_pt.pose.position.x = x[0]
        traj_full_state_pt.pose.position.y = x[1]
        traj_full_state_pt.pose.position.z = x[2]
        traj_full_state_pt.twist.linear.x = x[3]
        traj_full_state_pt.twist.linear.y = x[4]
        traj_full_state_pt.twist.linear.z = x[5]
        traj_full_state_pt.pose.orientation.w = x[6]
        traj_full_state_pt.pose.orientation.x = x[7]
        traj_full_state_pt.pose.orientation.y = x[8]
        traj_full_state_pt.pose.orientation.z = x[9]
        return traj_full_state_pt

    def x_u_2_traj_full_pt(self, x: np.ndarray, u: np.ndarray) -> TrajFullStatePt:
        traj_full_state_pt = self.x_2_full_state_pt(x)
        traj_full_state_pt.twist.angular.x = u[0]
        traj_full_state_pt.twist.angular.y = u[1]
        traj_full_state_pt.twist.angular.z = u[2]
        traj_full_state_pt.collective_force = u[3] * AP.mass
        return traj_full_state_pt


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

    # ROS convention, w > 0
    traj_full_state.pose.orientation.x = q_xyzw[0]
    traj_full_state.pose.orientation.y = q_xyzw[1]
    traj_full_state.pose.orientation.z = q_xyzw[2]
    traj_full_state.pose.orientation.w = q_xyzw[3]

    traj_full_state.twist.angular.x = p.item()
    traj_full_state.twist.angular.y = q.item()
    traj_full_state.twist.angular.z = r.item()

    traj_full_state.collective_force = u1

    return traj_full_state
