"""
This contains all relevant I/O operations.
"""

import json

import numpy as np
import pandas as pd
import pickle


def load_json(fpath, return_data="abstract", append_title=False):
    """ Loading the gathered json files """
    with open(fpath, encoding="utf-8") as f:
        data = json.load(f)
        data_df = pd.json_normalize(data["papers"])
        if return_data == "abstract":
            if append_title:
                data_df["abstract"] = data_df["title"] + " " + data_df["abstract"]
            corpus = data_df
        elif return_data == "keywords":
            corpus = data_df["keywords"]

    return corpus


def save_results(
    output_path, corpus, preprocessed_corpus, model, top_words_per_cluster
):
    corpus = pd.merge(corpus, preprocessed_corpus, on="abstract")
    results = {}
    for cluster in range(np.max(model.labels_) + 1):
        results[cluster] = {
            "top_words": top_words_per_cluster[cluster],
            "papers": list(corpus["title"][model.labels_ == cluster]),
        }

    with open(output_path, "w") as fp:
        json.dump(results, fp)


def save_model(output_path, model):
    with open(output_path, "wb") as fp:
        pickle.dump(model, fp)
