#! /usr/bin/env python3

import argparse
from enum import Enum
from pathlib import Path

import utils.data as io
import yaml
from sklearn.cluster import KMeans
from utils.clustering import Clustering
from utils.preprocessing import Preprocessing

"""
Load our collected papers as a json file, preprocess, perform vectorization
and cluster with selected algorithm.

We will configure this via yaml configs later.
"""

# TODO think about configuration


def get_parser():
    """ Parser to configure main script via command line """
    parser = argparse.ArgumentParser(description="scientific paper clustering")
    parser.add_argument("--option1", type=str, help="Example option")
    parser.add_argument(
        "--option2",
        type=int,
        default=1,
        help="Another option with typing",
    )
    return parser


class PreprocessingLib(Enum):
    NLTK = "nltk"
    SPACY = "spacy"


class DimReduction(Enum):
    LSA = "lsa"
    SPECTRAL = "spectral"
    NONE = "None"


class ClusteringModel(Enum):
    KMEANS = KMeans


# NOTE CHEN: I like to configure my experiments via yaml files
# The run.py script can be used to parse the configuration files to this file
if __name__ == "__main__":
    # Read YAML file
    with open("config.yaml") as stream:
        config = yaml.safe_load(stream)

    # Arguments
    input_path = config["input_path"]
    output_path = config["output_path"]
    use_title = config["use_title"]

    # preprocessing arguments
    preprocessing_lib = PreprocessingLib(config["preprocessing"]["lib"]).value
    use_stemming = config["preprocessing"]["stemming"]
    use_lemmatization = config["preprocessing"]["lemmatization"]
    min_word_len = config["preprocessing"]["min_word_len"]
    max_word_len = config["preprocessing"]["max_word_len"]
    custom_stopwords = config["preprocessing"]["custom_stopwords"]

    # dimensionality reduction
    dim_reduction = DimReduction(config["dimensionality_reduction"]).value

    # clustering
    model = (
        ClusteringModel.KMEANS.value
    )  # TODO: use value from config yaml ClusteringModel(config["clustering"]["model"]).value
    n_clusters = config["clustering"]["n_clusters"]

    # TODO: add Arg Parser

    #    global args, logger
    #   args = get_parser().parse_args()
    #    logger = helper.get_logger(args.save_folder)
    #    logger.info(args)
    #    logger.info("=> Starting new experiment ...")

    print("Load data")
    corpus = io.load_json(input_path, append_title=use_title)

    print("Perform preprocessing")
    preprocessed_corpus = Preprocessing(
        corpus["abstract"],
        lib=preprocessing_lib,
        stemming=use_stemming,
        lemmatization=use_lemmatization,
        min_word_len=min_word_len,
        max_word_len=max_word_len,
    ).apply_preprocessing()

    print("Start with clustering")
    clustering = Clustering(model, preprocessed_corpus, n_clusters)
    model = clustering.peform_clustering()
    top_words_per_cluster = clustering.get_top_words()

    print(f"Save results to {output_path}")
    io.save_results(
        output_path, corpus, preprocessed_corpus, model, top_words_per_cluster
    )
