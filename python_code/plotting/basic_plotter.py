from python_code.plotting.plotter_utils import get_deepsic, get_meta_deepsic, get_online_deepsic
from python_code.utils.config_singleton import Config
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.plotting.plotter_config import *
from python_code.trainers.trainer import Trainer
from dir_definitions import PLOTS_DIR, FIGURES_DIR
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os

conf = Config()


class Plotter:

    def __init__(self, run_over):
        self.run_over = run_over
        self.init_save_folder()

    def init_save_folder(self):
        # path for the saved figure
        current_day_time = datetime.datetime.now()
        self.folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
        if not os.path.isdir(os.path.join(FIGURES_DIR, self.folder_name)):
            os.makedirs(os.path.join(FIGURES_DIR, self.folder_name))

    def get_ber_plot(self, dec: Trainer, run_over: bool, method_name: str):
        print(method_name)
        # set the path to saved plot results for a single method (so we do not need to run anew each time)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_name = '_'.join([method_name,conf.train_frame_size,conf.SNR_start,conf.n_ecc_symbols])
        plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
        print(plots_path)
        # if plot already exists, and the run_over flag is false - load the saved plot
        if os.path.isfile(plots_path) and not run_over:
            print("Loading plots")
            ser_total = load_pkl(plots_path)
        else:
            # otherwise - run again
            print("calculating fresh")
            ser_total = dec.train()
            save_pkl(plots_path, ser_total)
        return ser_total

    def plot_ser(self, blocks_ind, ser, method_name):
        plt.plot(blocks_ind, np.cumsum(np.array(ser)) / len(ser),
                 label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name],
                 linewidth=2.2)
        plt.ylabel(r'BER', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        plt.grid(True, which='both')
        plt.yscale('log')
        plt.legend(loc='upper left', prop={'size': 15})

    def main(self, current_run_params):
        # get trainer
        trainer = current_run_params[0]
        # name of detector
        name = current_run_params[1]
        all_bers = self.get_ber_plot(trainer, run_over=self.run_over, method_name=name)
        self.plot_ser(range(conf.test_frame_num - 1), all_bers[0], name)
        plt.savefig(os.path.join(FIGURES_DIR, self.folder_name, f'SER_{conf.train_frame_size}.png'), bbox_inches='tight')


if __name__ == "__main__":
    plotter = Plotter(run_over=False)
    plotter.main(current_run_params=get_deepsic())
    plotter.main(current_run_params=get_online_deepsic())
    plotter.main(current_run_params=get_meta_deepsic())
    plt.show()
