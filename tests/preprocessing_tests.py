import unittest
import sys
import os

sys.path.append(os.path.abspath("../src"))
from utils.preprocessing import Preprocessing
from utils.config import PreprocessingConfig, PreprocessingLib
import utils.data as io
from pandas import DataFrame


class PreprocessingTest(unittest.TestCase):
    def __init__(self, *args):
        unittest.TestCase.__init__(self, *args)

        self.corpus = io.load_json(
            "../src/data/data_jmlr_vol13-21.json", append_title=False
        )

        self.config = PreprocessingConfig(
            PreprocessingLib.NLTK,
            False,
            True,
            2,
            15,
            [],
        )

    def test_preprocessing(self):
        """ Preprocessing returns corpus with correct shape and data types """

        preprocessed_corpus = Preprocessing(
            self.corpus["abstract"], config=self.config
        ).apply_preprocessing()

        self.assertTrue(type(preprocessed_corpus) is DataFrame)

        self.assertTrue(preprocessed_corpus.shape[1] == 2)  # two columns
        self.assertTrue(preprocessed_corpus.columns[0] == "abstract")
        self.assertTrue(preprocessed_corpus.columns[1] == "token")

        self.assertTrue(type(preprocessed_corpus.iloc[0][0]) == str)
        self.assertTrue(type(preprocessed_corpus.iloc[0][1]) == list)


if __name__ == "__main__":
    unittest.main()
