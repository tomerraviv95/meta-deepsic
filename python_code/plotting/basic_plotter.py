import os
import datetime

from dir_definitions import FIGURES_DIR
from python_code.plotting.plotter_utils import get_ber_plot, plot_ser
from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.utils.config_singleton import Config
import matplotlib.pyplot as plt

run_over = True
self_supervised_modes = [False,True]
trainer = DeepSICTrainer()
conf = Config()

# path for the saved figure
current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))

plt.figure()
for self_supervised in self_supervised_modes:
    conf.set_value('self_supervised', self_supervised)
    print(conf.self_supervised)
    all_bers = get_ber_plot(trainer, run_over=run_over, method_name=str(trainer))
    name = 'Online DeepSIC' if self_supervised else 'DeepSIC'
    plot_ser(range(conf.test_frame_num), all_bers[0], name)
plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER.png'), bbox_inches='tight')
plt.show()
