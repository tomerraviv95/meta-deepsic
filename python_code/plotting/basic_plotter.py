from python_code.trainers.trainer import Trainer
from python_code.utils.config_singleton import Config
from python_code.utils.python_utils import load_pkl, save_pkl
from python_code.plotting.plotter_config import *
from dir_definitions import PLOTS_DIR, FIGURES_DIR
import matplotlib.pyplot as plt
from typing import Tuple, List
import numpy as np
import datetime
import os

MARKER_EVERY = 5

conf = Config()


class Plotter:
    """
    Main Plotting function - for the figures of the paper, enter figure index below. It runs by applying the config
    from resources/config_runs with the same figure index.
    """

    def __init__(self, run_over: bool):
        self.run_over = run_over
        self.init_save_folder()

    def init_save_folder(self):
        # path for the saved figure
        current_day_time = datetime.datetime.now()
        self.folder_name = f'{current_day_time.month}-{current_day_time.day}-{current_day_time.hour}-{current_day_time.minute}'
        if not os.path.isdir(os.path.join(FIGURES_DIR, self.folder_name)):
            os.makedirs(os.path.join(FIGURES_DIR, self.folder_name))

    def get_ser_plot(self, trainer: Trainer, run_over: bool, method_name: str, trial: int = None):
        print(method_name)
        # set the path to saved plot results for a single method (so we do not need to run anew each time)
        if not os.path.exists(PLOTS_DIR):
            os.makedirs(PLOTS_DIR)
        file_name = '_'.join([method_name,
                              conf.channel_mode,
                              str(trainer.test_frame_size),
                              str(conf.info_size),
                              str(conf.snr), 'dB'])
        if not conf.linear_channel:
            file_name = file_name + '_non_linear'
        if trial is not None:
            file_name = file_name + '_' + str(trial)
        plots_path = os.path.join(PLOTS_DIR, file_name + '.pkl')
        print(plots_path)
        # if plot already exists, and the run_over flag is false - load the saved plot
        if os.path.isfile(plots_path) and not run_over:
            print("Loading plots")
            ser = load_pkl(plots_path)
        else:
            # otherwise - run again
            print("calculating fresh")
            ser = trainer.main()
            save_pkl(plots_path, ser)
        return ser

    def plot_ser_versus_block(self, blocks_ind: List[int], ser: List[float], method_name: str):
        plt.plot(blocks_ind, np.cumsum(np.array(ser)) / len(ser),
                 label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name],
                 linewidth=2.2,
                 markevery=MARKER_EVERY)
        plt.ylabel(r'Coded BER', fontsize=20)
        plt.xlabel(r'block index', fontsize=20)
        plt.yscale('log')
        plt.legend(loc='upper left', prop={'size': 15})

    def ser_versus_block(self, current_run_params: Tuple[Trainer, str]):
        # get trainer
        trainer = current_run_params[0]
        # name of detector
        name = current_run_params[1]
        ser = self.get_ser_plot(trainer, run_over=self.run_over, method_name=name)
        self.plot_ser_versus_block(range(conf.test_frame_num - 1), ser[0], name)
        plt.savefig(os.path.join(FIGURES_DIR, self.folder_name, f'SER_versus_block_{trainer.test_frame_size}.png'),
                    bbox_inches='tight')

    def plot_ser_versus_snr(self, blocks_ind: List[int], ser: List[float], method_name: str):
        plt.plot(blocks_ind, ser,
                 label=method_name,
                 color=COLORS_DICT[method_name],
                 marker=MARKERS_DICT[method_name],
                 linestyle=LINESTYLES_DICT[method_name],
                 linewidth=2.2)
        plt.ylabel(r'Coded BER', fontsize=20)
        plt.xlabel(r'SNR[dB]', fontsize=20)
        plt.grid(True, which='both')
        plt.yscale('log')
        plt.legend(loc='lower left', prop={'size': 15})

    def ser_versus_snr(self, current_run_params: Tuple[Trainer, str], trial_num: int):
        # get trainer
        trainer = current_run_params[0]
        # name of detector
        name = current_run_params[1]
        snr_values = list(range(11, 16))
        total_sers = []
        print(name)
        for snr in snr_values:
            conf.set_value('snr', snr)
            print(conf.snr)
            avg_ser = []
            for trial in range(trial_num):
                conf.set_value('seed', trial_num)
                trainer.__init__()
                ser_plot = self.get_ser_plot(trainer, run_over=self.run_over, method_name=name, trial=trial)
                cur_avg_ser = sum(ser_plot[0]) / len(ser_plot[0])
                avg_ser.append(cur_avg_ser)
            total_sers.append(np.mean(np.sort(np.array(avg_ser))))

        self.plot_ser_versus_snr(snr_values, total_sers, name)
        plt.savefig(os.path.join(FIGURES_DIR, self.folder_name, f'SER_versus_snr.png'), bbox_inches='tight')
