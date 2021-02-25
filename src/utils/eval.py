"""
Evaluation pipeline for evaluating clustering algorithms on research papers.
"""

import numpy as np
from sklearn import metrics


class Evaluation:
    """
    Wrapper class for sklearn metrics.
    """

    def __init__(self, features, corpus, logger):
        self.features = features
        self.prediction = corpus["predicted_labels"]
        self.logger = logger

        if "label" in corpus:
            self.groundtruth = corpus["label"]
        else:
            self.groundtruth = None

    def __len__(self):
        return len(self.prediction)

    def silhouette_score(self):
        return metrics.silhouette_score(
            self.features, self.prediction, metric="euclidean"
        )

    def calinski_harabasz(self):
        return metrics.calinski_harabasz_score(self.features, self.prediction)

    def davies_bouldin_score(self):
        return metrics.davies_bouldin_score(self.features, self.prediction)

    def purity(self):
        def purity_score(y_true, y_pred):
            """
            from this nice answer: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric
            """
            # compute contingency matrix (also called confusion matrix)
            contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
            # return purity
            return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(
                contingency_matrix
            )

        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return purity_score(self.groundtruth, self.prediction)

    def adjusted_rand_score(self):
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.adjusted_rand_score(self.groundtruth, self.prediction)

    def adjusted_mutual_info_score(self):
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.adjusted_mutual_info_score(self.groundtruth, self.prediction)

    def precision(self):
        """ NOTE: this can be dangerous as we may not have correct classifications, but just partitions. """
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.precision_score(
                self.groundtruth,
                self.prediction,
                average="weighted",
                zero_division=True,
            )

    def recall(self):
        """ NOTE: this can be dangerous as we may not have correct classifications, but just partitions. """
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.recall_score(
                self.groundtruth,
                self.prediction,
                average="weighted",
                zero_division=True,
            )

    def f1(self):
        """ NOTE: this can be dangerous as we may not have correct classifications, but just partitions. """
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.f1_score(
                self.groundtruth, self.prediction, average="weighted"
            )

    def accuracy(self):
        """ NOTE: this can be dangerous as we may not have correct classifications, but just partitions. """
        if self.groundtruth is None:
            raise Exception("Supervised metric needs groundtruth")
        else:
            assert len(self.groundtruth) == len(
                self.prediction
            ), "Groundtruth and predictions must have same length"
            return metrics.accuracy_score(self.groundtruth, self.prediction)

    def evaluate_all(self):
        all_scores = {}
        try:
            all_scores["silhouette_score"] = self.silhouette_score()
            all_scores["calinski_harabasz"] = self.calinski_harabasz()
            all_scores["davies_bouldin_score"] = self.davies_bouldin_score()
            if self.groundtruth is not None:
                all_scores["purity"] = self.purity()
                all_scores["adjusted_rand_score"] = self.adjusted_rand_score()
                all_scores[
                    "adjusted_mutual_info_score"
                ] = self.adjusted_mutual_info_score()
                all_scores["precision"] = self.precision()
                all_scores["recall"] = self.recall()
                all_scores["f1"] = self.f1()
                all_scores["accuracy"] = self.accuracy()
        except Exception as e:
            self.logger.error(f"Only one cluster found: {e}")
        return all_scores
