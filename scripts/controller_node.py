#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: controller_node.py
Date: 2023/4/15 上午10:33
Description:
"""

import sys
import os

current_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, current_path)

import copy
import time
import rospy
import actionlib

from mavros_msgs.msg import AttitudeTarget, State, ESCStatus, ESCStatusItem
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from oop_qd_onbd.msg import TrajCoefficients
from oop_qd_onbd.msg import TrackTrajAction, TrackTrajGoal, TrackTrajResult, TrackTrajFeedback


class ControllerNode:
    def __init__(self) -> None:
        rospy.init_node("tracking_controller", anonymous=False)
        qd_name = rospy.get_param(rospy.get_name() + "/qd_name")

        # Action
        self.tracking_server = actionlib.SimpleActionServer(
            "tracking_controller/track_traj", TrackTrajAction, self.track_traj_callback, auto_start=False
        )
        self.tracking_server.start()
        rospy.loginfo("Action Server started: tracking_controller/track_traj")

    def track_traj_callback(self, goal: TrackTrajGoal):
        rospy.loginfo("Receive a trajectory. Start tracking trajectory.")

        # TODO: tracking trajectory
        for i in range(10):
            time.sleep(0.5)

            # check for preempt
            if self.tracking_server.is_preempt_requested():
                rospy.loginfo("Trajectory tracking preempted.")
                self.tracking_server.set_preempted()
                return  # exit the callback and step into the next callback to handle new goal

            # publish feedback
            feedback = TrackTrajFeedback()
            feedback.percent_complete = i / 10
            feedback.tracking_error = Point(0.1 * i, 0.2 * i, 0.3 * i)
            rospy.loginfo(f"percent_complete: {feedback.percent_complete}")
            self.tracking_server.publish_feedback(feedback)

        rospy.loginfo("Trajectory tracking finished.")

        error_rmse = 1379  # the listener number on three-body plant
        self.tracking_server.set_succeeded(TrackTrajResult(error_rmse))


if __name__ == "__main__":
    try:
        node = ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
