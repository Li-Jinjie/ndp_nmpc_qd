import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import torch
import tqdm
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from nn_net import net
import argparse


def plot_histogram(data):
    # Define the bin edges (intervals)
    bin_edges = np.arange(0, 1.5, 0.1)
    hist, bins = np.histogram(data, bins=bin_edges)

    # Plot the histogram
    plt.bar(bins[:-1], hist, width=1, align="edge")  # Use width=1 for integer bins
    plt.xlabel("Interval")
    plt.ylabel("Frequency")
    plt.title("Histogram of Data within Intervals")
    plt.grid(axis="y", linestyle="--", alpha=0.7)


def plot_scatter(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots()
    ax.scatter(x, y, c="#00DDAA", marker="o", s=0.1, label=f"{xlabel}-{ylabel}")
    ax.set_xlabel(f"{xlabel} [m]")
    ax.set_ylabel(f"{ylabel} [m]")
    ax.set_title(title)
    ax.set_aspect(1)
    ax.grid()
    ax.legend()


def plot_3d_scatter(xs, ys, zs):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.scatter(xs, ys, zs, zdir="z", c="#00DDAA", marker="o", s=0.1)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()


def plot_pred_one_line(
    model, test_z_array=[-0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5], point_size=400, xy_range=1.0, color="viridis"
):
    fig, ax = plt.subplots(1, len(test_z_array), sharex=True, sharey=True, figsize=(6, 6))

    model.eval()
    model.to("cuda")
    test_f = []

    for k, test_z in enumerate(test_z_array):
        test_xy = np.linspace(start=-xy_range, stop=xy_range, num=point_size)
        test_matrix = np.zeros([point_size**2, 6])
        for i in range(point_size):
            for j in range(point_size):
                test_matrix[i * point_size + j, 0] = test_xy[i]
                test_matrix[i * point_size + j, 1] = test_xy[j]
                test_matrix[i * point_size + j, 2] = test_z
        torch_matrix = torch.from_numpy(test_matrix).to(torch.float32)
        input = torch.autograd.Variable(torch_matrix).cuda()
        output = model(input)
        test_f.append(output)

        zs = output[:, 2]
        plot_f = np.zeros((point_size, point_size))
        for i in range(point_size):
            for j in range(point_size):
                plot_f[i, j] = zs[i * point_size + j]

        im = ax[k].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        ax[k].set_title(f"{test_z}m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--SN", type=int, default=0, help="max spectral norm for a single layer")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")

    writer = SummaryWriter()

    options = {}
    options["epochs"] = parser.parse_args().epochs
    options["training_proportion"] = 0.75
    options["shuffle"] = True
    options["learning_r"] = 1e-4
    options["loss_type"] = "MSE"  # for regression
    options["SN"] = parser.parse_args().SN

    # ======== load data ==========
    # load data from csv
    input_df = pd.read_csv("downwash_input.csv")
    output_df = pd.read_csv("downwash_output.csv")

    x = input_df[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    y = output_df[["fx", "fy", "fz"]].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=options["training_proportion"], shuffle=options["shuffle"], random_state=42
    )

    x_train, y_train, x_test, y_test = [
        torch.from_numpy(data).to(torch.float32) for data in [x_train, y_train, x_test, y_test]
    ]

    # # ============= viz data =================
    # xs = x_train[:, 0]
    # ys = x_train[:, 1]
    # zs = x_train[:, 2]
    #
    # # histogram of zs
    # zs_valid = zs[xs**2 + ys**2 < 1.0**2]
    # plot_histogram(zs_valid)
    #
    # # scatter plot
    # plot_scatter(xs, ys, "x", "y", "x-y")
    # plot_scatter(ys, zs, "y", "z", "y-z")
    # plot_scatter(xs, zs, "x", "z", "x-z")
    #
    # plot_3d_scatter(x_train[:, 0], x_train[:, 1], x_train[:, 2])
    # plot_3d_scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2])
    #
    # plt.show()

    # ============= train ================
    net = net.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=options["learning_r"])
    num_epochs = options["epochs"]

    test_input = torch.autograd.Variable(x_test).cuda()
    test_target = torch.autograd.Variable(y_test).cuda()

    for epoch in tqdm.trange(num_epochs):
        # SN is not compatible with dropout
        net.train() if options["SN"] == 0 else net.eval()

        input = torch.autograd.Variable(x_train).cuda()
        target = torch.autograd.Variable(y_train).cuda()
        loss = criterion(net(input), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Spectral Normalization
        if options["SN"] > 0:
            for param in net.parameters():
                weight_mtx = param.detach()
                if weight_mtx.ndim > 1:
                    s = torch.linalg.norm(weight_mtx, 2)
                    # s = np.linalg.norm(M, 2)
                    if s > options["SN"]:
                        param.data = param / s * options["SN"]

        if (epoch + 1) % 10 == 0:
            writer.add_scalar(f"Loss/train/SN={options['SN']}", loss, epoch)

            net.eval()
            test_loss = criterion(net(test_input), test_target)
            writer.add_scalar(f"Loss/test/SN={options['SN']}", test_loss, epoch)

            if (epoch + 1) % 1000 == 0:
                print("epoch:", epoch + 1, " loss:", loss.item(), "test_loss: ", test_loss.item())

    test_loss = criterion(net(test_input), test_target)
    model_name = f"128-64-128_WBias_SN={options['SN']}_epoch={options['epochs']}_test_loss={round(test_loss.item(), 4)}"
    model_path = f"./nn_model/{model_name}.pkl"
    torch.save(net.state_dict(), model_path)

    writer.flush()
    writer.close()

    # figure 2
    plot_pred_one_line(net, [-0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
    plt.savefig(f"./imgs/{model_name}.pdf")

    # plt.show()
