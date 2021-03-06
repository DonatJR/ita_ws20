#! /usr/bin/env python3

import pandas as pd
import utils.data as io
from utils.clustering import Clustering
from utils.config import Config
from utils.eval import Evaluation
from utils.helper import get_logger, get_parser
from utils.preprocessing import Preprocessing

"""
Load our collected papers as a json file, preprocess, perform vectorization
and cluster with selected algorithm.
"""


def main():
    args = get_parser().parse_args()

    config = Config.from_file(args.config)

    logger = get_logger(config.output_path)
    logger.info("=> Starting new experiment ...")

    logger.info("Load data")
    corpus = io.load_json(config.input_path, append_title=config.use_title)

    logger.info("Perform preprocessing")
    corpus = corpus.drop_duplicates(subset="abstract")
    preprocessed_corpus = Preprocessing(
        corpus["abstract"],
        config=config.preprocessing,
        logger=logger,
    ).apply_preprocessing()

    logger.info("Start clustering")
    clustering = Clustering(
        preprocessed_corpus,
        clustering_config=config.clustering,
        dim_reduction_config=config.dim_reduction,
        logger=logger,
    )
    model, features = clustering.perform_clustering(return_features=True)
    top_words_per_cluster = clustering.get_top_words()

    preprocessed_corpus["predicted_labels"] = model.labels_
    preprocessed_corpus = pd.merge(corpus, preprocessed_corpus, on="abstract")

    logger.info(f"Save results to {config.output_path}")
    io.save_results(
        config.output_path + "cluster.json",
        preprocessed_corpus,
        top_words_per_cluster,
    )

    logger.info(f"Saving model to {config.output_path}")
    io.save_model(config.output_path + "model.pickle", model)

    if config.evaluate:
        logger.info("Evaluate Results")
        evaluations = Evaluation(
            features, preprocessed_corpus, logger=logger
        ).evaluate_all()
        io.save_evaluations(config.output_path + "evaluations.json", evaluations)


if __name__ == "__main__":
    main()
