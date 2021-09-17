from dir_definitions import CONFIG_RUNS_DIR
import os
from python_code.trainers.deepsic.joint_deep_sic_trainer import JointDeepSICTrainer
from python_code.trainers.deeprx.joint_deeprx_trainer import JointDeepRXTrainer
from python_code.trainers.deeprx.meta_deeprx_trainer import MetaDeepRXTrainer
from python_code.trainers.deepsic.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.trainers.deepsic.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.trainers.deeprx.online_deeprx_trainer import OnlineDeepRXTrainer
from python_code.utils.config_singleton import Config


def get_deepsic(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (JointDeepSICTrainer(), 'Joint DeepSIC')


def get_online_deepsic(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (OnlineDeepSICTrainer(), 'Online DeepSIC')


def get_online_deepsic_single_user(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (OnlineDeepSICTrainer(), 'Online DeepSIC - Single User')


def get_meta_deepsic(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC')


def get_meta_deepsic_single_user(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC - Single User')


def get_deeprx(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (JointDeepRXTrainer(), 'Joint DeepRX')


def get_online_deeprx(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (OnlineDeepRXTrainer(), 'Online DeepRX')


def get_meta_deeprx(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaDeepRXTrainer(), 'Meta-DeepRX')
