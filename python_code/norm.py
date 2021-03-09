from python_code.data.channel_model import ChannelModel
import matplotlib.pyplot as plt
import numpy as np

channel_mode = 'SED'
n_ant = 4
n_user = 4
H_0 = ChannelModel.calculate_channel_wrapper(channel_mode, n_ant, n_user)
norms = []
degs_array = np.array([51, 39, 33, 21]) / 5
for iteration in range(50):
    fade_mat = np.cos(np.deg2rad(iteration * degs_array))
    fade_mat = np.tile(fade_mat.reshape(1, -1), [n_user, 1])
    H = H_0 * fade_mat
    sub = H - H_0
    norm = np.linalg.norm(sub) ** 2
    norms.append(norm)

plt.plot(norms)
plt.show()
