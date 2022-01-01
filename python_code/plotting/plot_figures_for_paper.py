from python_code.plotting.plotter_utils import get_deepsic, get_online_deepsic, get_online_deepsic_single_user, \
    get_meta_deepsic, get_meta_deepsic_single_user, get_resnet, get_online_resnet, get_meta_resnet
from python_code.plotting.basic_plotter import Plotter
import matplotlib.pyplot as plt


# plot figures for the paper
def plot_figure_wrapper(figure_ind: int):
    if figure_ind in [1, 2, 5, 6, 7, 8, 11, 12, 15, 16, 17, 18, 23]:
        plotter.ser_versus_block(current_run_params=get_deepsic(figure_ind))
        if figure_ind in [7, 8, 17, 18, 23]:
            plotter.ser_versus_block(current_run_params=get_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_deepsic(figure_ind))
        if figure_ind in [7, 8, 17, 18, 23]:
            plotter.ser_versus_block(current_run_params=get_online_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_deepsic(figure_ind))
        if figure_ind in [7, 8, 17, 18, 23]:
            plotter.ser_versus_block(current_run_params=get_meta_resnet(figure_ind))
    if figure_ind in [3, 4, 13, 14]:
        plotter.ser_versus_block(current_run_params=get_online_deepsic(f'{figure_ind}a'))
        plotter.ser_versus_block(current_run_params=get_online_deepsic_single_user(f'{figure_ind}b'))
    if figure_ind in [5, 6, 15, 16]:
        plotter.ser_versus_block(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_deepsic_single_user(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_deepsic_single_user(figure_ind))
    if figure_ind in [9, 19]:
        plotter.ser_versus_blocks_num(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_online_deepsic(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_meta_deepsic(figure_ind))
    if figure_ind in [10, 20]:
        plotter.ser_versus_blocks_num(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_online_deepsic_single_user(figure_ind))
        plotter.ser_versus_blocks_num(current_run_params=get_meta_deepsic_single_user(figure_ind))
    if figure_ind in [21, 22, 24]:
        plotter.ser_versus_snr(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_resnet(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_online_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_online_resnet(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic(figure_ind))
        plotter.ser_versus_snr(current_run_params=get_meta_resnet(figure_ind))


if __name__ == "__main__":
    plotter = Plotter(run_over=False)
    figure_ind = 5
    plot_figure_wrapper(figure_ind)
    plt.show()
