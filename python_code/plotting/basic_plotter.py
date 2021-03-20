from python_code.plotting.plotter_utils import get_ber_plot, plot_ser
from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.utils.config_singleton import Config
from dir_definitions import FIGURES_DIR
import matplotlib.pyplot as plt
import datetime
import os

run_over = True

all_runs_params = [(DeepSICTrainer(), {'self_supervised': False, 'online_meta': False}, 'DeepSIC'),
                   (DeepSICTrainer(), {'self_supervised': True, 'online_meta': False}, 'Online DeepSIC'),
                   (MetaDeepSICTrainer(), {'self_supervised': True, 'online_meta': True}, 'Meta-DeepSIC')]
conf = Config()

# path for the saved figure
current_day_time = datetime.datetime.now()
folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
if not os.path.isdir(os.path.join(FIGURES_DIR, folder_name)):
    os.makedirs(os.path.join(FIGURES_DIR, folder_name))

plt.figure()
for current_run_params in all_runs_params:
    # get trainer
    trainer = current_run_params[0]
    # set all parameters based on dict
    for k, v in current_run_params[1].items():
        conf.set_value(k, v)
    # name of detector
    name = current_run_params[2]
    print(name)
    print(conf.self_supervised, conf.online_meta)
    all_bers = get_ber_plot(trainer, run_over=run_over, method_name=name)
    plot_ser(range(conf.test_frame_num), all_bers[0], name)
plt.savefig(os.path.join(FIGURES_DIR, folder_name, 'SER.png'), bbox_inches='tight')
plt.show()
