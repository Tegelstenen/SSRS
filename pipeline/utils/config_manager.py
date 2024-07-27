import yaml
from pathlib import Path

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            script_dir = Path(__file__).resolve().parent.parent.parent
            config_path = script_dir / 'config.yaml'
            with open(config_path, 'r') as config_file:
                cls._instance.config = yaml.safe_load(config_file)
        return cls._instance

    def get(self, key, default=None):
        return self.config.get(key, default)
    
# Usage in other scripts:
# from config_manager import ConfigManager
# config = ConfigManager()
# value = config.get('KEY')