from python_code.utils.python_utils import load_pkl, save_pkl
from dir_definitions import PLOTS_DIR
from python_code.plotting.plotter_config import *
from python_code.trainers.trainer import Trainer
import matplotlib.pyplot as plt
import os


def get_ser_plot(dec: Trainer, run_over: bool, method_name: str):
    print(method_name)
    # set the path to saved plot results for a single method (so we do not need to run anew each time)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    file_name = '_'.join([method_name])
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


def plot_ser(snr_list, ser, method_name):
    plt.figure()
    plt.plot(snr_list, ser,
             label=METHOD_NAMES[method_name],
             color=COLORS_DICT[method_name],
             marker=MARKERS_DICT[method_name],
             linestyle=LINESTYLES_DICT[method_name],
             linewidth=2.2)
    plt.ylabel(r'BER', fontsize=20)
    plt.xlabel(r'SNR [dB]', fontsize=20)
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.legend(loc='upper left', prop={'size': 15})
