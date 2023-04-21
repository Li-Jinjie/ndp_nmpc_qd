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

import time
import numpy as np
import rospy
import tf2_ros
import actionlib
from typing import List, Tuple

from mavros_msgs.msg import AttitudeTarget, State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion, Pose, PoseArray, TransformStamped
from oop_qd_onbd.msg import TrackTrajAction, TrackTrajGoal, TrackTrajResult, TrackTrajFeedback, PredXU

from nmpc_node import ControllerNode


class FollowerNode(ControllerNode):
    def __init__(self) -> None:
        super().__init__(
            has_traj_server=False, has_pred_viz=True, pred_viz_type="pred", is_build_acados=False, has_pred_pub=True
        )

        self.formation_ref = Point(x=1, y=1, z=0.5)
        rospy.Subscriber(f"/fhnp/traj_tracker/pred", PredXU, self.sub_pred_callback)
        rospy.Subscriber(f"{self.node_name}/formation_ref", Point, self.sub_formation_ref_callback)

    def sub_formation_ref_callback(self, msg: Point):
        self.formation_ref = msg

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


if __name__ == "__main__":
    try:
        node = FollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
