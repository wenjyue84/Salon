import json
import logging

CONFIG_FILE = 'config.json'

def load_config(config_file=CONFIG_FILE):
    try:
        with open(config_file, 'r') as file:
            config = json.load(file)
        required_keys = ['file_path', 'model']
        for key in required_keys:
            if key not in config:
                logging.error(f"Missing required configuration key: {key}")
                return None
        return config
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON configuration file: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error while loading configuration: {e}")
        return None
