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

    def get_ser_plot(self, trainer: Trainer, run_over: bool, method_name: str):
        print(method_name)
        # set the path to saved plot results for a single method (so we do not need to run anew each time)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_name = '_'.join([method_name,
                              conf.channel_mode,
                              str(trainer.total_frame_size),
                              str(conf.test_info_size),
                              str(conf.snr), 'dB'])
        plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
        print(plots_path)
        # if plot already exists, and the run_over flag is false - load the saved plot
        if os.path.isfile(plots_path) and not run_over:
            print("Loading plots")
            ser = load_pkl(plots_path)
        else:
            # otherwise - run again
            print("calculating fresh")
            ser = trainer.train()
            save_pkl(plots_path, ser)
        return ser

    def plot_ser_versus_block(self, blocks_ind, ser, method_name):
        plt.plot(blocks_ind, np.cumsum(np.array(ser)) / len(ser),
                 label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name],
                 linewidth=2.2)
        plt.ylabel(r'Coded BER', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        plt.grid(True, which='both')
        plt.yscale('log')
        plt.legend(loc='upper left', prop={'size': 15})

    def plot_ser_versus_blocks_num(self, blocks_ind, ser, method_name):
        plt.plot(blocks_ind, ser,
                 label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name],
                 linewidth=2.2)
        plt.ylabel(r'SER', fontsize=20)
        plt.xlabel(r'Data Block Size', fontsize=20)
        plt.grid(True, which='both')
        plt.yscale('log')
        plt.legend(loc='upper left', prop={'size': 15})

    def ser_versus_block(self, current_run_params):
        # get trainer
        trainer = current_run_params[0]
        # name of detector
        name = current_run_params[1]
        ser = self.get_ser_plot(trainer, run_over=self.run_over, method_name=name)
        self.plot_ser_versus_block(range(conf.test_frame_num - 1), ser[0], name)
        plt.savefig(os.path.join(FIGURES_DIR, self.folder_name, f'SER_versus_block_{trainer.total_frame_size}.png'),
                    bbox_inches='tight')

    def ser_versus_blocks_num(self, current_run_params):
        # get trainer
        trainer = current_run_params[0]
        # name of detector
        name = current_run_params[1]
        test_pilot_sizes = [20, 40, 60, 80, 100, 140, 180, 220, 250]
        data_frame_size = 300
        total_sers = []
        for test_pilot_size in test_pilot_sizes:
            conf.set_value('test_pilot_size', test_pilot_size)
            test_info_size = test_pilot_size + data_frame_size
            conf.set_value('test_info_size', test_info_size)
            trainer.__init__()
            ser = self.get_ser_plot(trainer, run_over=self.run_over, method_name=name)
            total_sers.append(sum(ser[0]) / len(ser[0]))

        self.plot_ser_versus_blocks_num(test_pilot_sizes, total_sers, name)
        plt.savefig(os.path.join(FIGURES_DIR, self.folder_name, f'SER_versus_pilot_size_{data_frame_size}_data.png'),
                    bbox_inches='tight')


if __name__ == "__main__":
    plotter = Plotter(run_over=False)
    plotter.ser_versus_blocks_num(current_run_params=get_deepsic())
    plotter.ser_versus_blocks_num(current_run_params=get_online_deepsic())
    plotter.ser_versus_blocks_num(current_run_params=get_meta_deepsic())
    plt.show()
