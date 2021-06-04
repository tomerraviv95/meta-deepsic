from python_code.trainers.joint_deep_sic_trainer import JointDeepSICTrainer
from python_code.trainers.online_deep_sic_trainer import OnlineDeepSICTrainer
from python_code.trainers.meta_deep_sic_trainer import MetaDeepSICTrainer
from python_code.utils.config_singleton import Config


def get_deepsic():
    return (JointDeepSICTrainer(), 'Joint DeepSIC')


def get_online_deepsic():
    return (OnlineDeepSICTrainer(), 'Online DeepSIC')


def get_online_deepsic_all_users():
    return (OnlineDeepSICTrainer(), 'Online DeepSIC - All Users')

def get_online_deepsic_user_dependent():
    conf = Config()
    user = 2
    conf.set_value('retrain_user', user)
    return (OnlineDeepSICTrainer(), f'Online DeepSIC - User {user}')


def get_meta_deepsic():
    return (MetaDeepSICTrainer(), 'Meta-DeepSIC')
