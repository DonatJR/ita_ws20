import os
import sys

if os.path.abspath("./src/") not in sys.path:
    sys.path.insert(0, os.path.abspath("./src/"))

if os.path.abspath("../src/") not in sys.path:
    sys.path.insert(0, os.path.abspath("../src/"))

import utils.clustering as clustering_module
import utils.preprocessing as preprocessing_module
import utils.data as data_module
import utils.config as config_module
import utils.helper as helper_module
