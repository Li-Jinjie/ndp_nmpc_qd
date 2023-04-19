import numpy as np
import matplotlib.pyplot as plt

# Define the data
epoch = np.array([1, 1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4])
train_loss = np.array([2.2060, 1.5871, 1.2323, 1.0925, 1.0254, 1.0183, 1.0198, 1.0207, 1.0204, 1.0241, 1.0290])
test_loss = np.array([2.1179, 1.5315, 1.2072, 1.0787, 1.0170, 1.0106, 1.0129, 1.0137, 1.0132, 1.0163, 1.0213])

plt.figure(figsize=(3.5, 2.5))
plt.plot(epoch, train_loss, "*-", label="Training Loss", color="blue")
plt.plot(epoch, test_loss, ".-", label="Test Loss", color="red")
plt.xlabel("Epoch", fontsize=8)
plt.ylabel("Loss", fontsize=8)
plt.tick_params(axis="both", which="major", labelsize=8)
plt.grid(True)
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.show()

plt.savefig("training_loss_for_disturb_nn.pdf", dpi=300, bbox_inches="tight")
