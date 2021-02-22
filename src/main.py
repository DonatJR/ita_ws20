#! /usr/bin/env python3

import argparse
from pathlib import Path

from utils.config import Config
import utils.data as io
from utils.helper import get_logger
from utils.clustering import Clustering
from utils.preprocessing import Preprocessing

"""
Load our collected papers as a json file, preprocess, perform vectorization
and cluster with selected algorithm.
"""


def get_parser():
    """ Parser to configure main script via command line """
    parser = argparse.ArgumentParser(description="scientific paper clustering")
    parser.add_argument(
        "--config", type=str, default="./config.yaml", help="Path to the config file"
    )
    return parser


def main():
    args = get_parser().parse_args()

    config = Config(args.config)

    logger = get_logger(config.output_path)
    logger.info(args)
    logger.info("=> Starting new experiment ...")

    logger.info("Load data")
    corpus = io.load_json(config.input_path, append_title=config.use_title)

    logger.info("Perform preprocessing")
    preprocessed_corpus = Preprocessing(
        corpus["abstract"], logger=logger, config=config.preprocessing
    ).apply_preprocessing()

    logger.info("Start clustering")
    clustering = Clustering(
        preprocessed_corpus,
        logger=logger,
        clustering_config=config.clustering,
        dim_reduction_config=config.dim_reduction,
    )
    model = clustering.perform_clustering()
    top_words_per_cluster = clustering.get_top_words()

    logger.info(f"Save results to {config.output_path}")
    io.save_results(
        config.output_path + "cluster.json",
        corpus,
        preprocessed_corpus,
        model,
        top_words_per_cluster,
    )

    logger.info(f"Saving model to {config.output_path}")
    io.save_model(config.output_path + "model.pickle", model)


if __name__ == "__main__":
    main()
