import os
import datetime

from dir_definitions import FIGURES_DIR
from python_code.plotting.plotter_utils import get_ser_plot, plot_ser
from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.utils.config_singleton import Config
import matplotlib.pyplot as plt

run_over = True
csi_noises = [0, 0.1]
trainer = DeepSICTrainer()
conf = Config()

# path for the saved figure
current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))

plt.figure()
for csi_noise in csi_noises:
    conf.set_value('csi_noise', csi_noise)
    print(conf.csi_noise)
    ser = get_ser_plot(trainer, run_over=run_over, method_name=str(trainer))
    plot_ser(conf.snr_list, ser, str(trainer))
plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER.png'), bbox_inches='tight')
plt.show()
