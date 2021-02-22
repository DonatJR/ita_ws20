import unittest
import sys
import os
import yaml

sys.path.append(os.path.abspath("../src"))
from utils.config import Config, ClusteringMethod, DimReduction, PreprocessingLib


class ConfigTest(unittest.TestCase):
    def test_config(self):
        """ Config parses yaml file correctly """

        yaml_dict = yaml.safe_load(
            """
            input_path: "input_path"
            output_path: "output_path"
            use_title: False
            preprocessing:
              stemming: False
              lemmatization: True
              lib: "nltk"
              min_word_len: 2
              max_word_len: 15
              custom_stopwords: ["test"]
            clustering:
              model: "agglomerative"
              n_clusters: 15
              agglomerative_linkage: "ward"
            embedding:
              dimensionality_reduction: "lsa"
              n_components: 2
        """
        )

        config = Config(yaml_dict)

        self.assertTrue(config.input_path == "input_path")
        self.assertTrue(config.output_path == "output_path")
        self.assertTrue(config.use_title == False)

        self.assertTrue(config.preprocessing.lib == PreprocessingLib.NLTK)
        self.assertTrue(config.preprocessing.use_stemming == False)
        self.assertTrue(config.preprocessing.use_lemmatization == True)
        self.assertTrue(config.preprocessing.min_word_len == 2)
        self.assertTrue(config.preprocessing.max_word_len == 15)
        self.assertTrue(len(config.preprocessing.custom_stopwords) == 1)
        self.assertTrue(config.preprocessing.custom_stopwords[0] == "test")

        self.assertTrue(config.dim_reduction.method == DimReduction.LSA)
        self.assertTrue(config.dim_reduction.n_components == 2)

        self.assertTrue(config.clustering.method == ClusteringMethod.AGGLOMERATIVE)
        self.assertTrue(config.clustering.n_clusters == 15)
        self.assertTrue(config.clustering.agglomerative_linkage == "ward")


if __name__ == "__main__":
    unittest.main()
