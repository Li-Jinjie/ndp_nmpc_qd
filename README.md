# OOP QD ONBD

OOP QD ONBD is a repository with an NMPC controller, a hover throttle estimator, and a downwash estimator. This repository receives traj info (TrajCoefficients.msg) from planner, and sends bodyrate cmd (using mavros_msg) to dop_qd_sim or any simulators or real quadrotors using mavros as the comm interface.

[TOC]

## Installation

1. Follow the installation in https://github.com/Li-Jinjie/dop_qd_sim
2. `git clone https://github.com/Li-Jinjie/oop_qd_onbd.git`
5. `catkin build` to build the whole workspace. Done!

## Getting Started

Before each running:  `cd /path_to_workspace` and then `source devel/setup.bash`

- If you want to make one quadrotor fly, just run `roslaunch oop_qd_onbd one_qd_nmpc.launch`
- If you want to make three quadrotor fly in a formation, just run `roslaunch oop_qd_onbd three_qd_nmpc_formation.launch`

Note that this repository is just a trajectory tracking controller, and the quadrotors remain hovering by default. If you want to make the quadrotor track a trajectory, please use `https://github.com/Li-Jinjie/cmd_pc` to send a trajectory cmd.

## License

GPLv3. Please also open-source your project if you use the code from this repository. Let's make the whole community better!
