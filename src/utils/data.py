"""
This contains all relevant I/O operations.
"""

import json
import pandas as pd


def load_json(fpath):
    """ Loading the manual json files prepared for the time being  """
    with open(fpath) as f:
        data = json.load(f)
        data_df = pd.json_normalize(data["papers"])
        corpus = data_df["abstract"]
    return corpus
