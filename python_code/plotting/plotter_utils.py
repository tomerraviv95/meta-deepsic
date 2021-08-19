from dir_definitions import CONFIG_RUNS_DIR
import os
from python_code.trainers.joint_deep_sic_trainer import JointDeepSICTrainer
from python_code.trainers.joint_deeprx_trainer import JointDeepRXTrainer
from python_code.trainers.meta_deeprx_trainer import MetaDeepRXTrainer
from python_code.trainers.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.trainers.online_deeprx_trainer import OnlineDeepRXTrainer
from python_code.utils.config_singleton import Config


def get_deepsic():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (JointDeepSICTrainer(), 'Joint DeepSIC')


def get_online_deepsic():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (OnlineDeepSICTrainer(), 'Online DeepSIC')


def get_meta_deepsic():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC')


def get_online_deepsic_single():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'single.yaml'))
    return (OnlineDeepSICTrainer(), 'Online DeepSIC - Single User Training')


def get_meta_deepsic_single():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'single.yaml'))
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC - Single User Training')


def get_deeprx():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (JointDeepRXTrainer(), 'Joint DeepRX')


def get_online_deeprx():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (OnlineDeepRXTrainer(), 'Online DeepRX')


def get_meta_deeprx():
    config = Config()
    config.load_config(os.path.join(CONFIG_RUNS_DIR, 'all.yaml'))
    return (MetaDeepRXTrainer(), 'Meta-DeepRX')
