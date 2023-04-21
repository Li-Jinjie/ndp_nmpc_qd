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
from oop_qd_onbd.msg import TrackTrajAction, TrackTrajGoal, TrackTrajResult, TrackTrajFeedback, MultiTrajFullStatePt

from nmpc_node import ControllerNode


class LeaderNode(ControllerNode):
    def __init__(self):
        super().__init__(has_traj_server=True, has_pred_viz=True, is_build_acados=False, has_pred_pub=True)

        self.formation_ref = Point(x=2, y=2, z=0.5)
        self.pub_formation_ref = rospy.Publisher(f"/xiao_feng/{self.node_name}/formation_ref", Point, queue_size=1)
        self.tmr_formation_ref = rospy.Timer(rospy.Duration(1 / 20), self.pub_formation_ref_callback)

    def pub_formation_ref_callback(self, timer: rospy.timer.TimerEvent):
        self.pub_formation_ref.publish(self.formation_ref)


if __name__ == "__main__":
    try:
        node = LeaderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
