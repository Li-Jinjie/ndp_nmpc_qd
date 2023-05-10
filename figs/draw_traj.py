#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Author: LI Jinjie
File: draw_traj.py
Date: 2023/5/3 下午8:43
Description:
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

# Load the data from the .csv file into a pandas DataFrame
traj_name = "dw_8"  # "fast_8" or "dw_8"
data = pd.read_csv(traj_name + ".csv")
data.index = data["__time"] - data["__time"][0]

# Set the font size for all elements
mpl.rcParams.update({"font.size": 14})


def format_y_tick_label(y, _):
    return f"{y:.1f}"


# Loop through the X, Y, and Z axes
for index, axis in enumerate(["x", "y", "z"]):
    # Create a new figure and axis
    fig, ax = plt.subplots()
    # Plot the reference data
    ref = data[f"/fhnp/traj_tracker/ref_x_u/x.0/data.{index}"].dropna()
    ref.plot(label="Reference", use_index=True, ax=ax, grid=True, color="#0072BD")
    # Plot the real data
    real = data[f"/fhnp/mavros/local_position/odom/pose/pose/position/{axis}"].dropna()
    real.plot(label="Real", use_index=True, ax=ax, grid=True, color="#D95319")
    # Set the axis labels
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel(f"Position {axis} (m)")
    # Make the curve fill the plot
    ax.autoscale(enable=True, axis="x", tight=True)
    # Add a legend to the plot
    ax.legend(fontsize=12)
    # Set the y-axis tick formatter to display one decimal place
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_label))
    # Save the figure
    plt.savefig(f"{traj_name}_p{axis}.svg", bbox_inches="tight")
    plt.show()

# Loop through the X, Y, and Z axes
for index, axis in enumerate(["x", "y", "z"]):
    # Create a new figure and axis
    fig, ax = plt.subplots()
    # Plot the reference data
    ref = data[f"/fhnp/traj_tracker/ref_x_u/x.0/data.{index+3}"].dropna()
    ref.plot(label="Reference", use_index=True, ax=ax, grid=True, color="#0072BD")
    # Plot the real data
    real = data[f"/fhnp/mavros/local_position/odom/twist/twist/linear/{axis}"].dropna()
    real.plot(label="Real", use_index=True, ax=ax, grid=True, color="#D95319")
    # Set the axis labels
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel(f"Velocity {axis} (m/s)")
    # Make the curve fill the plot
    ax.autoscale(enable=True, axis="x", tight=True)
    # Add a legend to the plot
    ax.legend(fontsize=12)
    # Set the y-axis tick formatter to display one decimal place
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_label))
    # Save the figure
    plt.savefig(f"{traj_name}_v{axis}.svg", bbox_inches="tight")
    plt.show()


# Loop through the X, Y, and Z axes
for index, axis in enumerate(["x", "y", "z"]):
    # Create a new figure and axis
    fig, ax = plt.subplots()
    # Plot the reference data
    ref = data[f"/fhnp/traj_tracker/ref_x_u/u.0/data.{index}"].dropna()
    ref.plot(label="Reference", use_index=True, ax=ax, grid=True, color="#0072BD")
    # Plot the real data
    real = data[f"/fhnp/mavros/local_position/odom/twist/twist/angular/{axis}"].dropna()
    real.plot(label="Real", use_index=True, ax=ax, grid=True, color="#D95319")
    # # Plot the Setpoint data
    # sp = data[f"/fhnp/mavros/setpoint_raw/attitude/body_rate/{axis}"].dropna()
    # sp.plot(label="Setpoint", use_index=True, ax=ax, grid=True, color="#EDB120")
    # Set the axis labels
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel(f"Body rate {axis} (rad/s)")
    # Make the curve fill the plot
    ax.autoscale(enable=True, axis="x", tight=True)
    # Add a legend to the plot
    ax.legend(fontsize=12)
    # Set the y-axis tick formatter to display one decimal place
    ax.yaxis.set_major_formatter(FuncFormatter(format_y_tick_label))
    # Save the figure
    plt.savefig(f"{traj_name}_w{axis}.svg", bbox_inches="tight")
    plt.show()
