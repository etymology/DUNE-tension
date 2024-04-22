import json
from typing import Dict, Optional

class ConfigManager:
    def __init__(self, filename: str = "config.json"):
        self.filename = filename
        self.config = self.load_config()

    def load_config(self) -> Dict:
        """Load configuration data from a JSON file."""
        try:
            with open(self.filename, "r") as config_file:
                config_data = json.load(config_file)
        except FileNotFoundError:
            print("Configuration file not found. Loading default settings.")
            config_data = self.default_config()
        return config_data

    def default_config(self) -> Dict:
        """Return the default configuration settings."""
        return {
            "current_apa": "Wood",
            "current_wirenumber": 0,
            "current_layer": "V",
            "sound_device_index": 0,
            "device_samplerate": 44100,
            "noise_threshold": 0.01
        }

    def save_config(self, config_data: Optional[Dict] = None):
        """Save the configuration data to a JSON file."""
        if config_data is not None:
            self.config = config_data
        with open(self.filename, "w") as config_file:
            json.dump(self.config, config_file)

    def update_config(self, key: str, value):
        """Update a specific configuration setting."""
        self.config[key] = value
       
