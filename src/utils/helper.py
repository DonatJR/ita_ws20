#! /usr/bin/env python3


"""
This is a collection of useful functions.
"""

import logging
import os
import pickle
import yaml
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler


def get_logger(save_path, name=None):
    """
    Create a log at specified path and addtional console logger.

    parameters:
    ---
    save_path [Path]: pathlib.Path object for saving
    """
    # Setup log file
    if name == None:
        logger_name = "main_logger.log"
    else:
        logger_name = name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        handler = TimedRotatingFileHandler(
            Path(save_path) / logger_name, interval=1, backupCount=0
        )
        logger.addHandler(handler)
        fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
        handler.setFormatter(logging.Formatter(fmt))

        # Add console output
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(fmt))
        logger.addHandler(console)

    return logger


# TODO: needed?
# NOTE this can be used in combination with argparse and yaml config files
# the execution is then carried out using two scripts
def execute_shell_script(command, config):
    """
    Execute shell script with config options using os.

    parameters:
    ---
    command [str]: /path/to/command
    config [dict]: Config dict usually from .yml file
    """
    shell_command = "python3 -u " + command + " "
    arg_dict = {}
    for key, value in config.items():
        if value != None:
            shell_command += "--" + key + "=" + str(value) + " "
        else:
            pass
    os.system(shell_command)
    return


def write_pickle(filepath, obj):
    """
    Use pickle to save data in a binary format.

    parameters:
    ---
    obj [dict]: Dictionary object
    filepath [pathlib.Path]: file path as Path object with .pkl as stem!
    """
    with open(str(filepath), "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    return


def read_pickle(filepath):
    """
    Use pickle to load data in a binary format.

    parameters:
    ---
    filepath [pathlib.Path]: file path
    """
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f)
    except EnvironmentError as err:
        print("Unable to open file: {}".format(err))
        sys.exit(1)


def read_yml(filepath):
    with open(filepath, "r") as ymlfile:
        try:
            data = yaml.safe_load(ymlfile)
        except yaml.YAMLError as exc:
            data = []
            print(exc)
            raise Exception("Failed to load yaml file!")
        finally:
            return data


def write_yml(data, path):
    try:
        with open(path, "w") as outfile:
            yaml.safe_dump(data, outfile, default_flow_style=False)
    finally:
        return
