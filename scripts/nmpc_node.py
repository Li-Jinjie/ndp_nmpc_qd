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
import actionlib

from mavros_msgs.msg import AttitudeTarget, State
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, PoseArray
from oop_qd_onbd.msg import TrackTrajAction, TrackTrajGoal, TrackTrajResult, TrackTrajFeedback

from pt_pub import NMPCRefPublisher
from nmpc import NMPCBodyRateController
from hv_throttle_est import HoverThrottleEstimator

from params import nmpc_params as CP, estimator_params as EP  # TODO: where is this CP should be?

# TODO: 1. estimation, 2. tf2_relay, 3. controller


class ControllerNode:
    def __init__(self) -> None:
        rospy.init_node("tracking_controller", anonymous=False)
        qd_name = rospy.get_param(rospy.get_name() + "/qd_name")

        # Action -> reference
        self.tracking_server = actionlib.SimpleActionServer(
            "tracking_controller/track_traj", TrackTrajAction, self.track_traj_callback, auto_start=False
        )
        self.tracking_server.start()
        rospy.loginfo("Action Server started: tracking_controller/track_traj")

        self.ref_pub = NMPCRefPublisher()

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
        self.nmpc_ctl = NMPCBodyRateController()
        self.nmpc_x_ref = np.zeros([CP.N_node + 1, CP.n_states])
        self.nmpc_u_ref = np.zeros([CP.N_node, CP.n_controls])
        self.tmr_control = rospy.Timer(rospy.Duration(CP.ts_nmpc), self.nmpc_callback)

        # - Estimator
        self.k_throttle = EP.k_throttle_init
        self.hv_th_estimator = HoverThrottleEstimator(EP.ts_est)
        self.tmr_hv_throttle_est = rospy.Timer(rospy.Duration(EP.ts_est), self.hover_throttle_callback)

        # Pub
        self.body_rate_cmd = AttitudeTarget()
        self.pub_attitude = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.pub_viz_pred = rospy.Publisher("tracking_controller/viz_pred", PoseArray, queue_size=10)

    def track_traj_callback(self, goal: TrackTrajGoal):
        """handle 3 task:
        1. receive a trajectory
        2. pub the trajectory reference points to the controller. give back the tracking error
        3. stop and restart the hover throttle estimation

        :param goal:
        :return:
        """
        rospy.loginfo("Receive a trajectory. Start tracking trajectory...")

        self.tmr_hv_throttle_est.shutdown()  # stop hover throttle estimation

        self.ref_pub.reset(goal.traj_coeff, rospy.Time.now())

        pos_rmse = 0
        yaw_rmse = 0

        while self.ref_pub.is_activated:
            # get reference
            # note that the pt_pub is asynchronous with controller, that is to say,
            # pt_pub doesn't wait for the controller to finish the previous step before publishing the next reference.
            self.nmpc_x_ref, self.nmpc_u_ref = self.ref_pub.get_nmpc_pts(rospy.Time.now())

            # check for preempt. Action related
            if self.tracking_server.is_preempt_requested():
                rospy.loginfo("Trajectory tracking preempted.")
                self.tracking_server.set_preempted()
                return  # exit the callback and step into the next callback to handle new goal

            # get error
            pos_err_now, yaw_err_now, pos_rmse, yaw_rmse = self.ref_pub.cum_error(self.px4_odom)

            # publish feedback
            feedback = TrackTrajFeedback()
            feedback.percent_complete = self.ref_pub.t_now / self.ref_pub.t_all
            feedback.pos_error = pos_err_now
            feedback.yaw_error = yaw_err_now
            rospy.loginfo(f"percent_complete: {feedback.percent_complete}")
            self.tracking_server.publish_feedback(feedback)

        rospy.loginfo("Trajectory tracking finished.")

        print(
            f"\n================================================\n"
            f"Positional error (RMSE): {pos_rmse:.6f} [m]\n"
            f"heading error (RMSE): {yaw_rmse:.6f} [deg]\n"
            f"================================================\n"
        )

        self.tmr_hv_throttle_est.start()  # restart hover throttle estimation

        self.tracking_server.set_succeeded(TrackTrajResult(pos_rmse, yaw_rmse))

    def nmpc_callback(self, timer: rospy.timer.TimerEvent):
        pass

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
