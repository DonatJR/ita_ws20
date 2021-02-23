#! /usr/bin/env python3

import utils.data as io
import itertools
from utils.clustering import Clustering
from utils.config import Config
from utils.helper import get_logger, get_parser
from utils.preprocessing import Preprocessing
import numpy as np

"""
Create a ground truth from clustering the keywords. This turned out to be the same
problem as before, i.e. it is hard to find meaningful words in these sets. Some
words are more informative than others and in the end we have a lot of noise that
makes paper appear very similar to each other.
The naive approach with binary vectors cannot resolve this, so we ran a dbscan on vectorized
preprocessed words.
"""

def flatten(list_of_lists):
    flattened = list(itertools.chain.from_iterable(list_of_lists))
    return flattened

def main():
    args = get_parser().parse_args()

    config = Config.from_file(args.config)

    logger = get_logger(config.output_path)
    logger.info(args)
    logger.info("=> Starting evaluation ...")

    logger.info("Load data")
    corpus = io.load_json(config.input_path, append_title=config.use_title)

    logger.info("Perform preprocessing")
    preprocessed_corpus = Preprocessing(
        corpus['keywords'],
        config=config.preprocessing,
        datatype='keywords',
        logger=logger,
    ).apply_preprocessing()

    preprocessed_corpus['token'] = preprocessed_corpus['token'].apply(flatten)
    preprocessed_corpus.drop('abstract', axis=1, inplace=True)

    logger.info("Start clustering")
    clustering = Clustering(
        preprocessed_corpus,
        clustering_config=config.clustering,
        dim_reduction_config=config.dim_reduction,
        logger=logger,
    )
    model = clustering.perform_clustering()

    # TODO: change output path in yaml to data
    print(np.unique(model.labels_))
    logger.info(f"Save results to {config.output_path}")
    corpus['label'] = model.labels_
    io.write_json(
        config.input_path + "labels.json",
        corpus
    )

if __name__ == "__main__":
    main()
