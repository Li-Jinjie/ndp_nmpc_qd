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

from pt_pub import NMPCRefPublisher
from nmpc import NMPCBodyRateController
from hv_throttle_est import HoverThrottleEstimator

from params import nmpc_params as CP, estimator_params as EP  # TODO: where is this CP should be?


class ControllerNode:
    def __init__(self) -> None:
        self.node_name = "traj_tracker"
        rospy.init_node(self.node_name, anonymous=False)

        self.namespace = rospy.get_namespace().rstrip("/")

        # Action -> reference
        self.pt_pub_server = actionlib.SimpleActionServer(
            f"{self.node_name}/pt_pub_action_server",
            TrackTrajAction,
            self.pt_pub_callback,
            auto_start=False,
        )

        self.ref_pub = NMPCRefPublisher()

        # Sub  -> feedback
        self.px4_state = State()
        self.px4_odom = None
        rospy.Subscriber(f"mavros/state", State, callback=self.sub_state_callback)
        rospy.Subscriber(f"mavros/local_position/odom", Odometry, self.sub_odom_callback)

        # Wait for Flight Controller connection
        rospy.loginfo(f"{self.namespace}: Waiting for the Flight Controller (eg. PX4) connection...")
        while not rospy.is_shutdown() and not self.px4_state.connected:
            time.sleep(0.5)
        rospy.loginfo(f"{self.namespace}: Flight Controller connected!")

        # Timer
        # - Controller
        self.nmpc_ctl = NMPCBodyRateController()
        self.nmpc_x_ref = np.zeros([CP.N_node + 1, CP.n_states])
        self.nmpc_u_ref = np.zeros([CP.N_node, CP.n_controls])
        while True:
            if self.px4_odom is not None:
                self.nmpc_x_ref, self.nmpc_u_ref = self.ref_pub.gen_fix_pt_ref(self.px4_odom)
                break
            time.sleep(0.2)

        self.tmr_control = rospy.Timer(rospy.Duration(CP.ts_nmpc), self.nmpc_callback)
        # self.tmr_pred_viz = rospy.Timer(rospy.Duration(0.05), self.viz_nmpc_pred_callback)

        # - Estimator
        self.k_throttle = EP.k_throttle_init
        self.hv_th_estimator = HoverThrottleEstimator(EP.ts_est)
        self.tmr_hv_throttle_est = rospy.Timer(rospy.Duration(EP.ts_est), self.hover_throttle_callback)

        # Pub
        self.body_rate_cmd = AttitudeTarget()
        self.pub_attitude = rospy.Publisher(f"mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.pub_viz_pred = rospy.Publisher(f"{self.node_name}/viz_pred", PoseArray, queue_size=10)
        self.pub_pred = rospy.Publisher(f"{self.node_name}/pred", MultiTrajFullStatePt, queue_size=10)

        # start action server after all the initialization is done
        self.pt_pub_server.start()
        rospy.loginfo(f"{self.namespace}: Action Server started: {self.node_name}/pt_pub_action_server")

    def pt_pub_callback(self, goal: TrackTrajGoal):
        """handle 3 task:
        1. receive a trajectory
        2. pub the trajectory reference points to the controller. give back the tracking error
        3. stop and restart the hover throttle estimation

        :param goal:
        :return:
        """
        rospy.loginfo(f"{self.namespace}: Receive a trajectory. Start tracking trajectory...")

        self.tmr_hv_throttle_est.shutdown()  # stop hover throttle estimation

        self.ref_pub.reset(goal.traj_coeff, rospy.Time.now())

        pos_rmse = 0
        yaw_rmse = 0

        r = rospy.Rate(1 / CP.ts_nmpc)
        while self.ref_pub.is_activated:
            # get reference
            # note that the pt_pub is asynchronous with controller, that is to say,
            # pt_pub doesn't wait for the controller to finish the previous step before publishing the next reference.
            self.nmpc_x_ref, self.nmpc_u_ref = self.ref_pub.get_nmpc_pts(rospy.Time.now())

            # check for preempt. Action related
            if self.pt_pub_server.is_preempt_requested():
                rospy.loginfo(f"{self.namespace}: Trajectory tracking preempted.")
                self.pt_pub_server.set_preempted()
                return  # exit the callback and step into the next callback to handle new goal

            # get error
            pos_err_now, yaw_err_now, pos_rmse, yaw_rmse = self.ref_pub.cum_error(self.px4_odom)

            # publish feedback
            feedback = TrackTrajFeedback()
            feedback.percent_complete = self.ref_pub.t_now / self.ref_pub.t_all
            feedback.pos_error = pos_err_now
            feedback.yaw_error = yaw_err_now
            rospy.loginfo_throttle(
                1, f"{self.namespace}: Trajectory tracking percent complete: {100 * feedback.percent_complete:.2f}%"
            )
            self.pt_pub_server.publish_feedback(feedback)

            r.sleep()

        rospy.loginfo(f"{self.namespace}: Trajectory tracking finished.")

        rospy.loginfo(
            f"{self.namespace}: \n"
            f"\n================================================\n"
            f"Positional error (RMSE): {pos_rmse:.6f} [m]\n"
            f"heading error (RMSE): {yaw_rmse:.6f} [deg]\n"
            f"================================================\n"
        )

        time.sleep(3)  # TODO: add safe check. only start next tracking when the qd reach the starting point

        # restart hover throttle estimation
        self.tmr_hv_throttle_est = rospy.Timer(rospy.Duration(EP.ts_est), self.hover_throttle_callback)

        self.pt_pub_server.set_succeeded(TrackTrajResult(pos_rmse, yaw_rmse))

    def nmpc_callback(self, timer: rospy.timer.TimerEvent):
        """NMPC controller callback
        only do one thing: track self.nmpc_x_ref and self.nmpc_u_ref
        """
        # ---- check if the control is too slow ----
        if timer.last_duration is not None and CP.ts_nmpc < timer.last_duration:
            rospy.logwarn(
                f"{self.namespace}: Control is too slow!"
                f"ts_ctl: {CP.ts_nmpc * 1000:.3f} ms < ts_one_round: {timer.last_duration * 1000:.3f} ms"
            )
        # ------------------------------------------

        nmpc_x0 = self.ref_pub.odom_2_nmpc_x(self.px4_odom)
        u0 = self.nmpc_ctl.update(nmpc_x0, self.nmpc_x_ref, self.nmpc_u_ref)
        self.body_rate_cmd = self.nmpc_u_2_att_tgt(u0[0], u0[1], u0[2], u0[3])
        self.pub_attitude.publish(self.body_rate_cmd)

        # for formation
        self.do_pub_pred()

    def do_pub_pred(self):
        # for formation
        mul_full_state_pts = MultiTrajFullStatePt()
        for i in range(self.nmpc_ctl.solver.N):
            x = self.nmpc_ctl.solver.get(i, "x")
            full_state_pt = self.ref_pub.x_2_full_state_pt(x)
            mul_full_state_pts.traj_pts.append(full_state_pt)

        mul_full_state_pts.header.stamp = rospy.Time.now()
        mul_full_state_pts.header.frame_id = "map"
        self.pub_pred.publish(mul_full_state_pts)

    def viz_nmpc_pred_callback(self, timer: rospy.timer.TimerEvent):
        flag = "pred"  # ref or pred
        viz_pred = PoseArray()
        for i in range(self.nmpc_ctl.solver.N):
            if flag == "ref":
                x = self.nmpc_x_ref[i]
            else:
                x = self.nmpc_ctl.solver.get(i, "x")

            p = Point(x=x[0], y=x[1], z=x[2])
            q = Quaternion(w=x[6], x=x[7], y=x[8], z=x[9])  # qw, qx, qy, qz
            pose = Pose(p, q)

            viz_pred.poses.append(pose)
        viz_pred.header.stamp = rospy.Time.now()
        viz_pred.header.frame_id = "map"
        self.pub_viz_pred.publish(viz_pred)

    def hover_throttle_callback(self, timer: rospy.timer.TimerEvent):
        vz = self.px4_odom.twist.twist.linear.z
        self.k_throttle, _, _ = self.hv_th_estimator.update(vz, self.body_rate_cmd.thrust)

    def sub_state_callback(self, msg: State):
        self.px4_state = msg

    def sub_odom_callback(self, msg: Odometry):
        self.px4_odom = msg  # note that qw < 0

        # tf2 pub
        br = tf2_ros.TransformBroadcaster()
        t = TransformStamped()

        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "map"
        t.child_frame_id = self.namespace + "/base_link"
        t.transform.translation = msg.pose.pose.position
        t.transform.rotation = msg.pose.pose.orientation

        br.sendTransform(t)

    def nmpc_u_2_att_tgt(self, rate_x, rate_y, rate_z, c):
        attitude_tgt = AttitudeTarget()

        attitude_tgt.type_mask = AttitudeTarget.IGNORE_ATTITUDE
        attitude_tgt.body_rate.x = rate_x
        attitude_tgt.body_rate.y = rate_y
        attitude_tgt.body_rate.z = rate_z

        attitude_tgt.thrust = c * CP.mass / self.k_throttle if self.k_throttle != 0 else 0  # throttle conversion

        return attitude_tgt


if __name__ == "__main__":
    try:
        node = ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass