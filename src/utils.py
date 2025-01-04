# src/utils.py
import os
from datasets import load_dataset
import configparser


def load_saved_dataset(dataset_name):
    """
    Loads the saved dataset from the specified directory.

    Args:
        dataset_name (str): The name of the dataset.
        data_dir (str): The directory where the dataset is saved.

    Returns:
        datasets.DatasetDict: Loaded dataset.
    """
    data_dir = 'data'
    # Load the dataset from the specified directory
    dataset = load_dataset('json', data_files={
        'train': os.path.join(data_dir, f"{dataset_name}_train.json"),
        'validation': os.path.join(data_dir, f"{dataset_name}_validation.json"),
        'test': os.path.join(data_dir, f"{dataset_name}_test.json")
    })
    
    return dataset


class read_config(config_file='config.ini'):
    """
    Reads the configuration file and returns the settings as a dictionary.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.
    """
    config = configparser.ConfigParser()
    
    # Check if the config file exists
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"The configuration file '{config_file}' does not exist.")
    
    config.read(config_file)
    
    # Convert config sections to a dictionary
    config_dict = {section: dict(config.items(section)) for section in config.sections()}
    
    return config_dict