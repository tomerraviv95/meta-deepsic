from dir_definitions import CONFIG_RUNS_DIR
import os
from python_code.trainers.deepsic.joint_deep_sic_trainer import JointDeepSICTrainer
from python_code.trainers.blackbox.joint_blackbox_trainer import JointBlackBoxTrainer
from python_code.trainers.blackbox.meta_blackbox_trainer import MetaBlackBoxTrainer
from python_code.trainers.deepsic.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.trainers.deepsic.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.trainers.blackbox.online_blackbox_trainer import OnlineBlackBoxTrainer
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
    return (OnlineDeepSICTrainer(), 'Online DeepSIC - Modular Training')


def get_meta_deepsic(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC')


def get_meta_deepsic_single_user(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC - Modular Training')


def get_resnet(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (JointBlackBoxTrainer(), 'Joint ResNet10')


def get_online_resnet(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (OnlineBlackBoxTrainer(), 'Online ResNet10')


def get_meta_resnet(figure_ind):
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, f'fig{figure_ind}.yaml'))
    return (MetaBlackBoxTrainer(), 'Meta-ResNet10')
