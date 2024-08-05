from ruamel.yaml import YAML
from pathlib import Path

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.script_dir = Path(__file__).resolve().parent.parent.parent
            cls._instance.config_path = cls._instance.script_dir / 'config.yaml'
            cls._instance.yaml = YAML()
            cls._instance.yaml.preserve_quotes = True
            cls._instance.yaml.indent(mapping=2, sequence=4, offset=2)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        with open(self.config_path, 'r') as config_file:
            self.config = self.yaml.load(config_file)

    def get(self, key, default=None):
        current = self.config
        for k in key.split('.'):
            if isinstance(current, dict):
                current = current.get(k)
            else:
                return default
            if current is None:
                return default
        return current

    def update_param(self, key, value):
        keys = key.split('.')
        current = self.config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

        # Write the updated config back to the file
        with open(self.config_path, 'w') as config_file:
            self.yaml.dump(self.config, config_file)

        # Reload the config
        self._load_config()

# Usage in other scripts:
# from config_manager import ConfigManager
# config = ConfigManager()
# value = config.get('KEY')
# config.update_param('KEY.SUBKEY', new_value)