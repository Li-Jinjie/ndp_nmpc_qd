import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import Axes3D
import torch
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from nn_net import Net

# load data from csv
input_df = pd.read_csv("downwash_input.csv")
output_df = pd.read_csv("downwash_output.csv")

input_pos = input_df[["x", "y", "z"]].to_numpy()
input_vel = input_df[["vx", "vy", "vz"]].to_numpy()
output_forces = output_df[["fx", "fy", "fz"]].to_numpy()
idx = np.expand_dims(np.arange(len(input_pos)), axis=1)

data_all = np.concatenate((idx, input_pos, input_vel, output_forces), axis=1)  # idx, x,y,z,vx,vy,vz,fx,fy,fz

options = {}
options["epochs"] = 10000
options["training_proportion"] = 0.7
options["shuffle"] = True
options["learning_r"] = 1e-4
options["loss_type"] = "MSE"  # "CrossEntropy"
options["SN"] = 4  # max spectral norm for a single layer

if options["shuffle"] is True:
    np.random.shuffle(data_all)

# 数据处理成网络输入
data_set_size = data_all.shape[0]
training_proportion = options["training_proportion"]
training_size = int(data_set_size * training_proportion)
test_size = data_set_size - training_size
x_training_np = data_all[0:training_size, 1:7]
x_test_np = data_all[training_size:data_set_size, 1:7]
y_training_np = data_all[0:training_size, 7:10]
y_test_np = data_all[training_size:data_set_size, 7:10]
x_training = torch.from_numpy(x_training_np).to(torch.float32)
y_training = torch.from_numpy(y_training_np).to(torch.float32)
x_test = torch.from_numpy(x_test_np).to(torch.float32)
y_test = torch.from_numpy(y_test_np).to(torch.float32)

# 数据分布绘图
xs = x_training[:, 0]
ys = x_training[:, 1]
zs = x_training[:, 2]

fig, ax = plt.subplots()
ax.scatter(xs, ys, c="#00DDAA", marker="o", s=0.1, label="x-y")  # Plot some data on the axes.
ax.set_xlabel("x [m]")  # Add an x-label to the axes.
ax.set_ylabel("y [m]")  # Add a y-label to the axes.
ax.set_title("x-y")  # Add a title to the axes.
ax.set_aspect(1)
ax.grid()
ax.legend()  # Add a legend.

fig, ax = plt.subplots()
ax.scatter(ys, zs, c="#00DDAA", marker="o", s=0.1, label="y-z")  # Plot some data on the axes.
ax.set_xlabel("y [m]")  # Add an x-label to the axes.
ax.set_ylabel("z [m]")  # Add a y-label to the axes.
ax.set_title("y-z")  # Add a title to the axes.
ax.set_aspect(1)
ax.grid()
ax.legend()  # Add a legend.

fig, ax = plt.subplots()
ax.scatter(xs, zs, c="#00DDAA", marker="o", s=0.1, label="x-z")  # Plot some data on the axes.
ax.set_xlabel("x [m]")  # Add an x-label to the axes.
ax.set_ylabel("z [m]")  # Add a y-label to the axes.
ax.set_title("x-z")  # Add a title to the axes.
ax.set_aspect(1)
ax.grid()
ax.legend()  # Add a legend.

fig = plt.figure()
ax = fig.gca(projection="3d")
xs = x_training[:, 0]
ys = x_training[:, 1]
zs = x_training[:, 2]
ax.scatter(xs, ys, zs, zdir="z", c="#00DDAA", marker="o", s=0.1)
ax.set(xlabel="X", ylabel="Y", zlabel="Z")
plt.show()


# training
model = Net().cuda()
if options["loss_type"] == "MSE":
    criterion = nn.MSELoss()
elif options["loss_type"] == "CrossEntropy":
    criterion = nn.CrossEntropyLoss()
else:
    raise ValueError("No this type of loss. Please check!")

optimizer = torch.optim.Adam(model.parameters(), lr=options["learning_r"])
num_epochs = options["epochs"]
test_input = torch.autograd.Variable(x_test).cuda()
test_target = torch.autograd.Variable(y_test).cuda()

test_out = model(test_input)
test_loss = criterion(test_out, test_target)
print("round: 0 test_loss: ", test_loss.item())

for epoch in tqdm.trange(num_epochs):
    input = torch.autograd.Variable(x_training).cuda()
    target = torch.autograd.Variable(y_training).cuda()
    out = model(input)
    loss = criterion(out, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Spectral Normalization
    if options["SN"] > 0:
        for param in model.parameters():
            weight_mtx = param.detach()
            if weight_mtx.ndim > 1:
                s = torch.linalg.norm(weight_mtx, 2)
                # s = np.linalg.norm(M, 2)
                if s > options["SN"]:
                    param.data = param / s * options["SN"]

    if (epoch + 1) % 1000 == 0:
        print("round:", epoch + 1, " loss:", loss.item())
        test_out = model(test_input)
        test_loss = criterion(test_out, test_target)
        print("test_loss: ", test_loss.item())

# test on test set
test_input = torch.autograd.Variable(x_test).cuda()
test_target = torch.autograd.Variable(y_test).cuda()
test_out = model(test_input)
test_loss = criterion(test_out, test_target)
print("test_loss: ", test_loss.item())

model_path = (
    "./nn_model/WBias_SN="
    + str(options["SN"])
    + "_epoch="
    + str(options["epochs"])
    + "_test_loss="
    + str(round(test_loss.item(), 4))
    + ".pkl"
)
torch.save(model.state_dict(), model_path)  # 保存模型
