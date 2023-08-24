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
from params import downwash_params as DP, nmpc_params as CP

from nmpc_node import ControllerNode
from dnwash_nn_est import DownwashNN

from ndp_nmpc import NDPNMPCBodyRateController


class NDPLeaderNode(ControllerNode):
    def __init__(self):
        super().__init__(
            NMPC_CTL=NDPNMPCBodyRateController,
            has_traj_server=True,
            has_pred_viz=True,
            is_build_acados=True,
            has_pred_pub=True,
        )

        # nn downwash observer
        self.downwash_observer = DownwashNN()
        self.sub_pred = rospy.Subscriber(f"/xiao_feng/traj_tracker/ref_x_u", PredXU, self.sub_xf_pred_callback)

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

    def sub_xf_pred_callback(self, msg: PredXU):
        # make traj target
        x_list = msg.x
        nmpc_x_other = np.zeros(shape=self.nmpc_x_ref.shape, dtype=np.float64)

        ego_position = self.px4_odom.pose.pose.position
        if (msg.x[0].data[0] - ego_position.x) ** 2 + (
            msg.x[0].data[1] - ego_position.y
        ) ** 2 < DP.r_horiz**2:  # take effect
            for i in range(len(x_list)):
                x = np.array(x_list[i].data, dtype=np.float64)
                nmpc_x_other[i] = x

            # make prediction
            self.disturb_force = self.downwash_observer.update(nmpc_x_other, self.nmpc_x_ref)
        else:
            self.disturb_force = np.zeros([len(x_list), 3])


if __name__ == "__main__":
    try:
        node = NDPLeaderNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
