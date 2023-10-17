# NDP NMPC QD

NDP NMPC QD is a repository with three packages: cmd_pc, ndp_nmpc, and dop_sim.

The main package is ndp_nmpc, including an NMPC controller, a hover throttle estimator, and a downwash estimator. This repository receives traj. info (TrajCoefficients.msg) from cmd_pc (planner), and then sends bodyrate cmd (using mavros_msg) to dop_sim or any simulators or real quadrotors. Communication between packages leverages mavros as the comm interface.

Read our paper for more details: https://arxiv.org/abs/2304.07794.

## Citation

```
@INPROCEEDINGS{
  author={Li, Jinjie and Han, Liang and Yu, Haoyang and Lin, Yuheng and Li, Qingdong and Ren, Zhang},
  booktitle={62nd IEEE Conference on Decision and Control (CDC)}, 
  title={Nonlinear MPC for Quadrotors in Close-Proximity Flight with Neural Network Downwash Prediction}, 
  year={2023},
  pages={1-7}}
```

## Installation

1. Follow the installation in https://github.com/Li-Jinjie/dop_sim
2. `git clone https://github.com/Li-Jinjie/ndp_nmpc_qd.git`
3. `catkin build` to build the whole workspace.
4. install acados

Done!

## Getting Started

Before each running:  `cd /path_to_workspace` and then `source devel/setup.bash`

- If you want to make one quadrotor fly, just run `roslaunch ndp_nmpc_qd one_qd_nmpc.launch`
- Then run `rosrun cmd_pc planner_node.py ` to send a trajectory!

Instead,

- If you want to make three quadrotor fly in a formation, just run `roslaunch ndp_nmpc three_qd_nmpc_formation.launch`

## Workflow

![](./UMLs/workflow.svg)

## License

GPLv3. Please also open-source your project if you use the code from this repository. Let's make the whole community better!
