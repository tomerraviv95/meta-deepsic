from dir_definitions import CONFIG_PATH
import yaml
import os


class Config:
    __instance = None

    def __new__(cls):
        if Config.__instance is None:
            Config.__instance = object.__new__(cls)
            Config.__instance.config = None
            Config.__instance.load_default_config()
        return Config.__instance

    def load_default_config(self):
        self.load_config(CONFIG_PATH)

    def load_config(self, config_path):
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.config_name = os.path.splitext(os.path.basename(config_path))[0]

        # set attribute of Trainer with every config item
        for k, v in self.config.items():
            setattr(self, k, v)

    def set_value(self, field, value):
        setattr(self, field, value)
