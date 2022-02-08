from python_code.plotting.plotter_utils import get_deepsic, get_online_deepsic, get_online_deepsic_single_user, \
    get_meta_deepsic, get_meta_deepsic_single_user, get_resnet, get_online_resnet, get_meta_resnet
from python_code.plotting.basic_plotter import Plotter
import matplotlib.pyplot as plt


# plot figures for the paper
def plot_figure_wrapper(figure_ind: int):
    if figure_ind in [1, 3, 5]:
        plotter.ser_versus_block(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_resnet(figure_ind))
    if figure_ind in [2, 4, 6]:
        plotter.ser_versus_snr(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_resnet(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_online_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_online_resnet(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_meta_resnet(figure_ind))
    if figure_ind in [7, 8]:
        plotter.ser_versus_snr(current_run_params=get_deepsic(f'{figure_ind}a'))
        plotter.ser_versus_snr(current_run_params=get_online_deepsic(f'{figure_ind}a'))
        plotter.ser_versus_snr(current_run_params=get_online_deepsic_single_user(f'{figure_ind}b'))
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic(f'{figure_ind}a'))
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic_single_user(f'{figure_ind}b'))
    if figure_ind in [9, 10]:
        plotter.ser_versus_blocks_num(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_online_deepsic_single_user(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_meta_deepsic_single_user(figure_ind))


if __name__ == "__main__":
    plotter = Plotter(run_over=False)
    figure_ind = 7
    plot_figure_wrapper(figure_ind)
    plt.show()
