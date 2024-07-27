from utils.config_manager import ConfigManager
import os


class HelperFunctions:
    config = ConfigManager()

    @classmethod
    def get_data_folder(cls) -> str:
        return os.path.join(cls.config.get('BASE_DIR'), cls.config.get('TEMP_DATA_DIR'))