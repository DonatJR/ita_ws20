import unittest
import os

from sklearn.cluster import KMeans

from context import clustering_module, preprocessing_module, data_module, config_module


class ClusteringTest(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)

        data_path = (
            "../src/data/data_jmlr_vol13-21.json"
            if os.path.abspath(".").endswith("ita_ws20\\tests")
            else "./src/data/data_jmlr_vol13-21.json"
        )
        corpus = data_module.load_json(data_path, append_title=False)

        config = config_module.PreprocessingConfig(
            config_module.PreprocessingLib.NLTK,
            False,
            True,
            2,
            15,
            [],
        )

        self.corpus = preprocessing_module.Preprocessing(
            corpus["abstract"], config=config
        ).apply_preprocessing()

    def test_model_type(self):
        """ Clustering returns the correct model type """

        clustering_config = config_module.ClusteringConfig(
            config_module.ClusteringMethod.KMEANS, 15, None
        )
        dim_reduction_config = config_module.DimReductionConfig(
            config_module.DimReduction.NONE, 2
        )

        model = clustering_module.Clustering(
            self.corpus, clustering_config, dim_reduction_config
        ).perform_clustering()
        self.assertTrue(type(model) is KMeans)

    def test_top_words_shape(self):
        """ Clustering returns top words in correct shape """

        clustering_config = config_module.ClusteringConfig(
            config_module.ClusteringMethod.KMEANS, 15, None
        )
        dim_reduction_config = config_module.DimReductionConfig(
            config_module.DimReduction.NONE, 2
        )

        clustering = clustering_module.Clustering(
            self.corpus, clustering_config, dim_reduction_config
        )
        clustering.perform_clustering()
        top_words = clustering.get_top_words()

        self.assertTrue(len(top_words) == 15)
        self.assertTrue(len(top_words[0]) == 10)


if __name__ == "__main__":
    unittest.main()
