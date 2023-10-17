import torch.nn as nn

# dropout_1 = 0.1
# dropout_2 = 0.2
# dropout_3 = 0.3

net = nn.Sequential(
    nn.Linear(6, 128),
    nn.ReLU(),
    # nn.Dropout(dropout_1),
    nn.Linear(128, 64),
    nn.ReLU(),
    # nn.Dropout(dropout_2),
    nn.Linear(64, 128),
    nn.ReLU(),
    # nn.Dropout(dropout_3),
    nn.Linear(128, 3),
)
