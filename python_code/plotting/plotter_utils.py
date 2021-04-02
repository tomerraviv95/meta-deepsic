from python_code.trainers.deep_sic_trainer import DeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer


def get_deepsic():
    return (DeepSICTrainer(), {'self_supervised': False, 'online_meta': False}, 'DeepSIC')


def get_online_deepsic():
    return (DeepSICTrainer(), {'self_supervised': True, 'online_meta': False}, 'Online DeepSIC')


def get_meta_deepsic():
    return (MetaDeepSICTrainer(), {'self_supervised': True, 'online_meta': True}, 'Meta-DeepSIC')
