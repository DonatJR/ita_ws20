"""
This contains all relevant I/O operations.
"""

import json

import numpy as np
import pandas as pd
from utils.helper import write_pickle


def write_json(fpath, data):
    new_dict = {"papers": data.to_dict(orient="records")}
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(new_dict, f, ensure_ascii=False, indent=4)


def load_json(fpath, return_data="abstract", append_title=False):
    """ Loading the gathered json files """
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
        data_df = pd.json_normalize(data["papers"])
        if return_data == "abstract":
            if append_title:
                data_df["abstract"] = data_df["title"] + " " + data_df["abstract"]
            corpus = data_df
        elif return_data == "keywords":
            corpus = data_df["keywords"]

    return corpus


def save_results(output_path, corpus, top_words_per_cluster):
    results = {}
    for cluster in range(np.max(corpus["predicted_labels"]) + 1):
        results[cluster] = {
            "top_words": top_words_per_cluster[cluster],
            "papers": list(corpus["title"][corpus["predicted_labels"] == cluster]),
        }

    with open(output_path, "w") as fp:
        json.dump(results, fp)


def save_model(output_path, model):
    write_pickle(output_path, model)


def save_evaluations(output_path, evaluations):
    with open(output_path, "w") as fp:
        json.dump(evaluations, fp)
