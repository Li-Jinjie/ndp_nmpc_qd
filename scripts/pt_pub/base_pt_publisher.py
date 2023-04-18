#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: minimum_snap.py
Date: 2022/5/9 17:36
LastEditors: LI Jinjie
LastEditTime: 2022/5/9 17:36
Description: trajectory generator, using closed-form Minimum Snap method.
"""
import numpy as np
import rospy
import tf_conversions

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from oop_qd_onbd.msg import TrajCoefficients, TrajPt  # TODO: change ROS msg to python msg
from .polym_optimizer import PolymOptimizer, MinMethod


class BasePtPublisher:
    def __init__(self, xyz_method=MinMethod.SNAP, yaw_method=MinMethod.ACCEL) -> None:
        self.optr_x = PolymOptimizer(xyz_method)
        self.optr_y = PolymOptimizer(xyz_method)
        self.optr_z = PolymOptimizer(xyz_method)
        self.optr_yaw = PolymOptimizer(yaw_method)

        self.traj_coeff = TrajCoefficients()

        # states need reset
        self.tracking_pos_err = 0.0  # m
        self.tracking_yaw_err = 0.0  # deg
        self.tracking_pt_num = 0.0
        self.start_ros_t = rospy.Time.now()  # store start time in rospy.Time
        self.is_activated = False  # if finish the whole trajectory, False.
        self.t_all = 0.0

        # states now
        self.t_now = 0.0
        self.traj_pt_now = TrajPt()

    def reset(self, traj_coeff: TrajCoefficients, ros_t: rospy.Time) -> None:
        self.traj_coeff = traj_coeff

        self.tracking_pos_err = 0.0  # m
        self.tracking_yaw_err = 0.0  # deg
        self.tracking_pt_num = 0.0
        self.start_ros_t = ros_t
        self.is_activated = True
        self.t_all = traj_coeff.traj_time_cum[-1]

    def cum_error(self, odom_now: Odometry) -> (float, float, float, float):
        """cumulate RMSE error. print the final result after the traj is finished."""

        # calculate error. error = target - now
        traj_pt = self.traj_pt_now
        pos_now = odom_now.pose.pose.position

        pos_err = (
            (traj_pt.position.x - pos_now.x) ** 2
            + (traj_pt.position.y - pos_now.y) ** 2
            + (traj_pt.position.z - pos_now.z) ** 2
        )

        q = odom_now.pose.pose.orientation
        euler = tf_conversions.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        yaw_now = euler[2]
        yaw_err = (np.degrees(traj_pt.yaw) - np.degrees(yaw_now)) ** 2

        self.tracking_pos_err += pos_err
        self.tracking_yaw_err += yaw_err
        self.tracking_pt_num += 1

        return (
            pos_err,
            yaw_err,
            np.sqrt(self.tracking_pos_err / self.tracking_pt_num),  # rmse for pos
            np.sqrt(self.tracking_yaw_err / self.tracking_pt_num),  # rmse for yaw
        )

    def get_pt(self, ros_t: rospy.Time, is_pred: bool = False) -> TrajPt:
        """
        get the trajectory point at time t.
        :param ros_t: time using rospy.Time.now(
        :param is_pred: is this a prediction point? if True, will not change self.is_activate.
        :return:
        """
        traj_pt = TrajPt()

        t = (ros_t - self.start_ros_t).to_sec()

        # Finish Detection
        if t >= self.traj_coeff.traj_time_cum[-1]:  # change to "hover" after finished
            traj_pt.position = self.traj_coeff.final_pt
            if not is_pred:
                self.is_activated = False
            return traj_pt

        # Generate Pt
        time_cum = np.array(self.traj_coeff.traj_time_cum)
        time_seg = np.array(self.traj_coeff.traj_time_seg)
        t_index = np.argwhere(time_cum > t)[0].item() - 1  # t_index ? S_i

        t_segment = time_seg[t_index]
        t_scaled = (t - time_cum[t_index]) / t_segment

        # - x,y,z
        c_x = _get_specific_coeff(t_index, self.optr_x, np.array(self.traj_coeff.coeff_x))
        c_y = _get_specific_coeff(t_index, self.optr_y, np.array(self.traj_coeff.coeff_y))
        c_z = _get_specific_coeff(t_index, self.optr_z, np.array(self.traj_coeff.coeff_z))

        traj_pt.position.x = _get_output_value(self.optr_x, 0, t_scaled, t_segment, c_x)
        traj_pt.position.y = _get_output_value(self.optr_y, 0, t_scaled, t_segment, c_y)
        traj_pt.position.z = _get_output_value(self.optr_z, 0, t_scaled, t_segment, c_z)

        traj_pt.velocity.x = _get_output_value(self.optr_x, 1, t_scaled, t_segment, c_x)
        traj_pt.velocity.y = _get_output_value(self.optr_y, 1, t_scaled, t_segment, c_y)
        traj_pt.velocity.z = _get_output_value(self.optr_z, 1, t_scaled, t_segment, c_z)

        traj_pt.accel.x = _get_output_value(self.optr_x, 2, t_scaled, t_segment, c_x)
        traj_pt.accel.y = _get_output_value(self.optr_y, 2, t_scaled, t_segment, c_y)
        traj_pt.accel.z = _get_output_value(self.optr_z, 2, t_scaled, t_segment, c_z)

        traj_pt.jerk.x = _get_output_value(self.optr_x, 3, t_scaled, t_segment, c_x)
        traj_pt.jerk.y = _get_output_value(self.optr_y, 3, t_scaled, t_segment, c_y)
        traj_pt.jerk.z = _get_output_value(self.optr_z, 3, t_scaled, t_segment, c_z)

        # - yaw
        c_yaw = _get_specific_coeff(t_index, self.optr_yaw, np.array(self.traj_coeff.coeff_yaw))
        traj_pt.yaw = _get_output_value(self.optr_yaw, 0, t_scaled, t_segment, c_yaw)
        traj_pt.yaw_dot = _get_output_value(self.optr_yaw, 1, t_scaled, t_segment, c_yaw)

        # Update States Now
        if not is_pred:
            self.t_now = t
            self.traj_pt_now = traj_pt

        return traj_pt


def _get_output_value(optr: PolymOptimizer, deriv: int, t_scaled: float, t_segment: float, coeff: np.array) -> float:
    """get the output value under the specific deriv order, time, and coefficient. scale the time here

    Returns:
        real output value: float
    """
    poly_time_d = optr.get_poly_params(deriv, t_scaled) / (np.power(t_segment, deriv))
    return (poly_time_d @ coeff).item()


def _get_specific_coeff(t_index: int, optimizer: PolymOptimizer, coeff_all: np.array) -> np.array:
    """get polynomial parameters in this segment from all coefficients

    Args:
        t_index: which segment in waypoints
        optimizer: polynomial optimizer
        coeff_all: all parameters computed before

    Returns:
        polynomial parameters in this segment

    """
    if coeff_all.ndim == 1:
        coeff_all = coeff_all[:, np.newaxis]

    n = optimizer.ord_polym
    row_shift = t_index * (n + 1)
    return coeff_all[row_shift : row_shift + (n + 1), :]
