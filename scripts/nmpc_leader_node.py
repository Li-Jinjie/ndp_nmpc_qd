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
from oop_qd_onbd.msg import PredXU
from params import downwash_params as DP

from nmpc_node import ControllerNode
from dnwash_nn_est import DownwashNN


class LeaderNode(ControllerNode):
    def __init__(self):
        super().__init__(
            has_traj_server=True, has_pred_viz=True, pred_viz_type="ref", is_build_acados=False, has_pred_pub=True
        )

        # formation target
        self.xf_formation_ref = Point(x=0.0, y=1.0, z=1.0)
        self.pub_xf_formation_ref = rospy.Publisher(f"/xiao_feng/{self.node_name}/formation_ref", Point, queue_size=1)
        self.sb_formation_ref = Point(x=0.0, y=-1.0, z=1.0)
        self.pub_sb_formation_ref = rospy.Publisher(f"/smile_boy/{self.node_name}/formation_ref", Point, queue_size=1)
        self.tmr_formation_ref = rospy.Timer(rospy.Duration(1 / 20), self.pub_formation_ref_callback)

    def pub_formation_ref_callback(self, timer: rospy.timer.TimerEvent):
        if np.abs(self.px4_odom.pose.pose.position.x - 1) > 2:
            self.xf_formation_ref = Point(x=0.0, y=0.0, z=0.5)
            self.sb_formation_ref = Point(x=0.0, y=-1.0, z=0.0)
        else:
            self.xf_formation_ref = Point(x=0.0, y=1.0, z=0.0)
            self.sb_formation_ref = Point(x=0.0, y=-1.0, z=0.0)

        self.pub_xf_formation_ref.publish(self.xf_formation_ref)
        self.pub_sb_formation_ref.publish(self.sb_formation_ref)


if __name__ == "__main__":
    try:
        node = LeaderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
