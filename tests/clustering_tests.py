import unittest
from sklearn.cluster import KMeans

from utils.clustering import Clustering
from utils.preprocessing import Preprocessing
from utils.config import (
    ClusteringConfig,
    DimReductionConfig,
    PreprocessingConfig,
    ClusteringMethod,
    DimReduction,
    PreprocessingLib,
)
import utils.data as io


class ClusteringTest(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)

        corpus = io.load_json("../src/data/data_jmlr_vol13-21.json", append_title=False)

        config = PreprocessingConfig(
            PreprocessingLib.NLTK,
            False,
            True,
            2,
            15,
            [],
        )

        self.corpus = Preprocessing(
            corpus["abstract"], config=config
        ).apply_preprocessing()

    def test_model_type(self):
        """ Clustering returns the correct model type """

        clustering_config = ClusteringConfig(ClusteringMethod.KMEANS, 15, None)
        dim_reduction_config = DimReductionConfig(DimReduction.NONE, 2)

        model = Clustering(
            self.corpus, clustering_config, dim_reduction_config
        ).perform_clustering()
        self.assertTrue(type(model) is KMeans)

    def test_top_words_shape(self):
        """ Clustering returns top words in correct shape """

        clustering_config = ClusteringConfig(ClusteringMethod.KMEANS, 15, None)
        dim_reduction_config = DimReductionConfig(DimReduction.NONE, 2)

        clustering = Clustering(self.corpus, clustering_config, dim_reduction_config)
        clustering.perform_clustering()
        top_words = clustering.get_top_words()

        self.assertTrue(len(top_words) == 15)
        self.assertTrue(len(top_words[0]) == 10)


if __name__ == "__main__":
    unittest.main()
