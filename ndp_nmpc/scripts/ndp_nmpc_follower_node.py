#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: nmpc_node.py
Date: 2023/4/15 10:33 AM
Description:
"""

import sys
import os

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)

import numpy as np
import rospy

from geometry_msgs.msg import Point
from ndp_nmpc.msg import PredXU

from nmpc_node import ControllerNode
from hv_throttle_est import AlphaFilter


class NDPFollowerNode(ControllerNode):
    def __init__(self, is_print_error=False) -> None:
        super().__init__(has_traj_server=False, has_pred_viz=True, is_build_acados=False, has_pred_pub=True)

        rospy.Subscriber(f"/fhnp/traj_tracker/pred", PredXU, self.sub_pred_callback)

        self.formation_ref = Point(x=1, y=1, z=0.5)
        self.lpf_ref_alpha = 0.8
        self.lpf_form_ref_x, self.lpf_form_ref_y, self.lpf_form_ref_z = None, None, None
        rospy.Subscriber(f"{self.node_name}/formation_ref", Point, self.sub_formation_ref_callback)

        self.is_print_error = is_print_error
        if self.is_print_error:
            self.num_pt = 0
            self.form_x_error_2 = 0
            self.form_y_error_2 = 0
            self.form_z_error_2 = 0

    def sub_formation_ref_callback(self, msg: Point):

        if self.lpf_form_ref_x is None:
            self.lpf_form_ref_x = AlphaFilter(alpha=self.lpf_ref_alpha, y0=msg.x)

        if self.lpf_form_ref_y is None:
            self.lpf_form_ref_y = AlphaFilter(alpha=self.lpf_ref_alpha, y0=msg.y)

        if self.lpf_form_ref_z is None:
            self.lpf_form_ref_z = AlphaFilter(alpha=self.lpf_ref_alpha, y0=msg.z)

        self.formation_ref.x = self.lpf_form_ref_x.update(msg.x)
        self.formation_ref.y = self.lpf_form_ref_y.update(msg.y)
        self.formation_ref.z = self.lpf_form_ref_z.update(msg.z)

    def sub_pred_callback(self, msg: PredXU):
        # make traj target
        x_list = msg.x
        u_list = msg.u

        for i in range(len(x_list)):
            x = np.array(x_list[i].data, dtype=np.float64)

            x[0] += self.formation_ref.x
            x[1] += self.formation_ref.y
            x[2] += self.formation_ref.z

            self.nmpc_x_ref[i] = x

            if i != (len(x_list) - 1):
                u = np.array(u_list[i].data, dtype=np.float64)
                self.nmpc_u_ref[i] = u

        if self.is_print_error:
            self.calculate_form_error()

    def calculate_form_error(self):
        self.num_pt += 1
        ego = self.px4_odom.pose.pose.position
        self.form_x_error_2 += (ego.x - self.nmpc_x_ref[0, 0]) ** 2
        self.form_y_error_2 += (ego.y - self.nmpc_x_ref[0, 1]) ** 2
        self.form_z_error_2 += (ego.z - self.nmpc_x_ref[0, 2]) ** 2

        rospy.loginfo_throttle(
            0.1,
            f"{self.namespace}\n"
            f"formation_x_error [RMSE]: {np.sqrt(self.form_x_error_2 / self.num_pt)}, \n"
            f"formation_y_error [RMSE]: {np.sqrt(self.form_y_error_2 / self.num_pt)}, \n"
            f"formation_z_error [RMSE]: {np.sqrt(self.form_z_error_2 / self.num_pt)}, \n"
            f"formation_error [RMSE]: "
            f"{np.sqrt((self.form_x_error_2 + self.form_y_error_2 + self.form_z_error_2) / self.num_pt)}",
        )


if __name__ == "__main__":
    try:
        node = NDPFollowerNode(is_print_error=False)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
