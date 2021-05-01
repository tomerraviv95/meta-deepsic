from python_code.trainers.joint_deep_sic_trainer import JointDeepSICTrainer
from python_code.trainers.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer


def get_deepsic():
    return (JointDeepSICTrainer(), 'Joint DeepSIC')


def get_online_deepsic():
    return (OnlineDeepSICTrainer(), 'Online DeepSIC')


def get_meta_deepsic():
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC')
