#!/usr/bin/env python3

import os
import yaml
import ipdb
import datetime
import shutil
import argparse

import utils.helper as helper

"""
Use this script to execute main.py with options from config.yaml. 
The config will be saved at a timestamped folder for saving along results.
This allows us to keep track of our experiments
"""


def main():
    """ Execute script with specified yaml parameters """
    config = helper.read_yml("config.yml")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_save_folder_name = "test_run_" + str(timestamp)
    config["output_dir"] = os.path.join(config["output_dir"], new_save_folder_name)
    os.makedirs(config["output_dir"], exist_ok=False)
    shutil.copy2("config.yml", os.path.join(config["output_dir"], "test_config.yml"))
    helper.execute_shell_script("main.py", config)


if __name__ == "__main__":
    main()
