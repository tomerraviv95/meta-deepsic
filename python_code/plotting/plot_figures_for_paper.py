from python_code.plotting.plotter_utils import get_deepsic, get_online_deepsic, get_online_deepsic_single_user, \
    get_meta_deepsic, get_meta_deepsic_single_user, get_resnet, get_online_resnet, get_meta_resnet
from python_code.plotting.basic_plotter import Plotter
import matplotlib.pyplot as plt

trials_num_dict = {2: 10,
                   4: 10,
                   6: 10,
                   7: 20,
                   8: 80}


# plot figures for the paper
def plot_figure_wrapper(figure_ind: int):
    if figure_ind in [1, 3, 5]:
        plotter.ser_versus_block(current_run_params=get_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_online_resnet(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_deepsic(figure_ind))
        plotter.ser_versus_block(current_run_params=get_meta_resnet(figure_ind))
    elif figure_ind in [2, 4, 6]:
        trial_num = trials_num_dict[figure_ind]
        plotter.ser_versus_snr(current_run_params=get_deepsic(figure_ind), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_resnet(figure_ind), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_online_deepsic(figure_ind), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_online_resnet(figure_ind), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic(figure_ind), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_meta_resnet(figure_ind), trial_num=trial_num)
    elif figure_ind in [7, 8]:
        trial_num = trials_num_dict[figure_ind]
        plotter.ser_versus_snr(current_run_params=get_deepsic(f'{figure_ind}a'), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_online_deepsic(f'{figure_ind}a'), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_online_deepsic_single_user(f'{figure_ind}b'), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic(f'{figure_ind}a'), trial_num=trial_num)
        plotter.ser_versus_snr(current_run_params=get_meta_deepsic_single_user(f'{figure_ind}b'), trial_num=trial_num)
    else:
        raise ValueError("No such figure index!!!")


if __name__ == "__main__":
    plotter = Plotter(run_over=False)
    figure_ind = 8
    plot_figure_wrapper(figure_ind)
    plt.show()
