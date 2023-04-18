#!/usr/bin/env python
# -*- encoding: ascii -*-
"""
Author: LI Jinjie
File: controller_node.py
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
import actionlib

from mavros_msgs.msg import AttitudeTarget, State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseArray
from oop_qd_onbd.msg import TrackTrajAction, TrackTrajGoal, TrackTrajResult, TrackTrajFeedback

from nmpc import NMPCBodyRateController
from estimator import HoverThrottleEstimator

from params import nmpc_params as CP, estimator_params as EP  # TODO: where is this CP should be?

# TODO: 1. estimation, 2. tf2_relay, 3. controller


class ControllerNode:
    def __init__(self) -> None:
        rospy.init_node("tracking_controller", anonymous=False)
        qd_name = rospy.get_param(rospy.get_name() + "/qd_name")

        # # NMPC
        # self.nmpc_opt = NMPCBodyRateController()

        # Action -> reference
        self.tracking_server = actionlib.SimpleActionServer(
            "tracking_controller/track_traj", TrackTrajAction, self.track_traj_callback, auto_start=False
        )
        self.tracking_server.start()
        rospy.loginfo("Action Server started: tracking_controller/track_traj")

        # Sub  -> feedback
        self.px4_state = State()
        self.px4_odom = Odometry()
        rospy.Subscriber("mavros/state", State, callback=self.sub_state_callback)
        rospy.Subscriber("mavros/local_position/odom", Odometry, self.sub_odom_callback)

        # Wait for Flight Controller connection
        while not rospy.is_shutdown() and not self.px4_state.connected:
            time.sleep(0.5)
        rospy.loginfo("Flight Controller connected!")

        # Timer
        # - Controller
        # self.tmr_control = rospy.Timer(rospy.Duration(0.1), self.control_callback)
        # - Estimator
        self.k_throttle = EP.k_throttle_init
        self.hv_th_estimator = HoverThrottleEstimator(EP.ts_est)
        self.tmr_hv_throttle_est = rospy.Timer(rospy.Duration(EP.ts_est), self.hover_throttle_callback)

        # Pub
        self.body_rate_cmd = AttitudeTarget()
        self.pub_attitude = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.pub_viz_pred = rospy.Publisher("tracking_controller/viz_pred", PoseArray, queue_size=10)

    def track_traj_callback(self, goal: TrackTrajGoal):
        rospy.loginfo("Receive a trajectory. Start tracking trajectory.")

        self.tmr_hv_throttle_est.shutdown()  # stop hover throttle estimation

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

        self.tmr_hv_throttle_est.start()  # restart hover throttle estimation

        self.tracking_server.set_succeeded(TrackTrajResult(error_rmse))

    def hover_throttle_callback(self, timer: rospy.timer.TimerEvent):
        vz = self.px4_odom.twist.twist.linear.z
        self.k_throttle, _, _ = self.hv_th_estimator.update(vz, self.body_rate_cmd.thrust)

    def sub_state_callback(self, msg: State):
        self.px4_state = msg

    def sub_odom_callback(self, msg: Odometry):
        self.px4_odom = msg


if __name__ == "__main__":
    try:
        node = ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
