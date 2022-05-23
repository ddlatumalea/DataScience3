"""Module that reads yaml and returns the paths of directories."""

from pathlib import Path
import yaml

def read_yaml():
    """Read the configuration file and load it into config."""
    with open('config.yaml', 'r') as stream:
        config = yaml.safe_load(stream)

    return config

def get_dir(key):
    """Get the path of a directory.
    
    Keyword arguments:
    key -- the name of the path
    """
    config = read_yaml()

    return Path(config[key])

def get_data_dir():
    """Returns the data directory"""
    return get_dir('data_dir')

def get_img_dir():
    """Returns the image directory"""
    return get_dir('img_dir')

def get_model_saves_dir():
    """Returns the directory where models are saved"""
    return get_dir('model_saves_dir')

def get_results_dir():
    """Returns the directory of the performance of models"""
    return get_dir('results_dir')