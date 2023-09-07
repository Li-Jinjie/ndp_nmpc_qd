import numpy as np
import matplotlib.pyplot as plt

# Define the data
data = np.load("/home/lijinjie/ljj_ws/src/ndp_nmpc_qd/figs/hv_est.npy")

plt.figure(figsize=(3.5, 2.5))
plt.plot(data[:, 0], data[:, 1], "-", color="#0072BD")
plt.xlabel("time t [s]", fontsize=8)
plt.ylabel("gamma", fontsize=8)
plt.tick_params(axis="both", which="major", labelsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("hover_estimator.svg", bbox_inches="tight")
plt.show()
