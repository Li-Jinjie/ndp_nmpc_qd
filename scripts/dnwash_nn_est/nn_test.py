import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import torch
from nn_net import net


def show_forces(num_test_f, z_value, point_size, xy_range):
    fig, ax = plt.subplots(1, 3, sharex=True)
    # fig, ax = plt.subplots(1, 3, figsize=(13, 3))
    fig.suptitle(f"z={z_value} m")

    color = "viridis"

    # x
    zs = num_test_f[:, 0]
    plot_f = np.zeros((point_size, point_size))
    for i in range(point_size):
        for j in range(point_size):
            plot_f[i, j] = zs[i * point_size + j]
    im = ax[0].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
    ax[0].set_xlabel("y [m]")
    ax[0].set_ylabel("x [m]")
    ax[0].set_title("Force x")
    ax[0].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax[0].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    # y
    zs = num_test_f[:, 1]
    plot_f = np.zeros((point_size, point_size))
    for i in range(point_size):
        for j in range(point_size):
            plot_f[i, j] = zs[i * point_size + j]
    im = ax[1].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
    ax[1].set_xlabel("y [m]")
    # ax[1].set_ylabel("x [m]")
    ax[1].set_title("Force y")
    ax[1].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax[1].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    # z
    zs = num_test_f[:, 2]
    plot_f = np.zeros((point_size, point_size))
    for i in range(point_size):
        for j in range(point_size):
            plot_f[i, j] = zs[i * point_size + j]

    im = ax[2].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
    ax[2].set_xlabel("y [m]")
    # ax[2].set_ylabel("x [m]")
    ax[2].set_title("Force z")
    ax[2].yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax[2].xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

    plt.tight_layout()

    fig.subplots_adjust(right=0.88)

    cbar_ax = fig.add_axes([0.92, 0.365, 0.02, 0.27])
    fig.colorbar(im, cax=cbar_ax)

    # if not os.path.exists("./imgs/" + model_name):
    #     os.mkdir("./imgs/" + model_name)
    # plt.savefig("./imgs/" + model_name + f"/z={z_value}.pdf")

    plt.show()


if __name__ == "__main__":
    model_path = "./nn_model/128-64-128_WBias_SN=4_epoch=10000_test_loss=1.0017.pkl"
    model = net
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cuda")

    point_size = 200
    xy_range = 1.0  # meter

    test_z_array = [-1.3, -1.1, -0.9, -0.7, -0.5, 0, 0.4, 0.5, 0.6, 0.7, 0.9, 1.1, 1.3]
    test_f = []

    for test_z in test_z_array:
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

    fig_f = plt.figure()
    ax = fig_f.gca(projection="3d")
    xs = test_matrix[:, 0]
    ys = test_matrix[:, 1]
    num_test = test_f[7].cpu().detach().numpy()
    # zs = np.linalg.norm(num_test_f,axis=1)
    zs = num_test[:, 2]
    ax.scatter(xs, ys, zs, zdir="z", c="#00DDAA", marker="o", s=0.1)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    plt.show()

    # for i, test_z in enumerate(test_z_array):
    #     num_test_f_1 = test_f[i].cpu().detach().numpy()
    #     show_forces(num_test_f_1, z_value=test_z, point_size=point_size, xy_range=xy_range)

    fig, ax = plt.subplots(3, 7, sharex=True, sharey=True, figsize=(6, 6))
    # 全局设置
    point_size = 400
    xy_range = 1.0  # meter
    color = "viridis"
    test_z_array = [-0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

    # No SN
    model_name = "128-64-128_WBias_SN=0_epoch=20000_test_loss=0.5181.pkl"
    model.load_state_dict(torch.load("./nn_model/" + model_name))
    # model.load_state_dict(torch.load("./model/model_params.pt"))
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

        im = ax[2][k].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)

    # SN=4
    model_name = "128-64-128_WBias_SN=4_epoch=20000_test_loss=1.0221.pkl"
    model.load_state_dict(torch.load("./nn_model/" + model_name))
    # model.load_state_dict(torch.load("./model/model_params.pt"))
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

        im = ax[1][k].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)

    # SN=2
    model_name = "128-64-128_WBias_SN=2_epoch=20000_test_loss=1.5888.pkl"
    model.load_state_dict(torch.load("./nn_model/" + model_name))
    # model.load_state_dict(torch.load("./model/model_params.pt"))
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

        im = ax[0][k].imshow(plot_f, vmin=-6, vmax=2, extent=[-xy_range, xy_range, xy_range, -xy_range], cmap=color)
        if k == 6:
            ax[0][k].set_title(f"z={test_z}m")
        else:
            ax[0][k].set_title(f"{test_z}")

    ax[0][0].set_ylabel(f"γ=2\nL=1.589")
    ax[1][0].set_ylabel(f"γ=4\nL=1.022")
    ax[2][0].set_ylabel(f"γ=∞ (No γ)\nLoss=0.518")

    # colorbar
    fig.subplots_adjust(wspace=0.3, hspace=-0.75, right=0.86)
    cbar_ax = fig.add_axes([0.88, 0.30, 0.020, 0.38])
    cbar = fig.colorbar(im, cax=cbar_ax)  # orientation="horizontal" if want horizontal
    cbar.ax.set_ylabel(f"Predicted force along z axis [N]", rotation=90)
    plt.savefig("./imgs/disturb_pred_horiz_new.pdf")
    plt.show()
