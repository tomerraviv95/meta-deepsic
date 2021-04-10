from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer


def get_deepsic():
    return (DeepSICTrainer(), {'self_supervised': False}, 'DeepSIC')


def get_online_deepsic():
    return (DeepSICTrainer(), {'self_supervised': True}, 'Online DeepSIC')


def get_meta_deepsic():
    return (MetaDeepSICTrainer(), {'self_supervised': True}, 'Meta-DeepSIC')
