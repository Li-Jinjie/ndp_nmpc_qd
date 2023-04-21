# -*- encoding: ascii -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # full_connected
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x


class DownwashNN:
    def __init__(self):
        self.device = torch.device("cuda")
        self.nn_model = Net()
        self.nn_model.load_state_dict(
            torch.load("../dnwash_nn_est/nn_model/bias0.686_SN=4_epoch=20000_test_loss=1.0527_model_params_bebest.pkl")
        )
        self.nn_model.to(self.device)
        self.nn_model.eval()  # set to evaluation mode
        self.nn_model = torch.jit.script(self.nn_model)  # jit acceleration

    def update(self, other_pred_x: np.array, ego_pred_x: np.array):
        input_np = (other_pred_x - ego_pred_x)[:, 0:6]
        input_torch = torch.from_numpy(input_np).to(torch.float32)
        input_cuda = input_torch.to(self.device)

        output = self.nn_model(input_cuda)

        output_np = output.cpu().detach().numpy()
        return output_np
