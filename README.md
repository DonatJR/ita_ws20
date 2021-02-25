# Scientific paper clustering - ITA WS20/21

## Team members

* Daniela Fichiu
  * 3552717
  * BSc Applied Computer Science & BSc Mathematics
  * daniela.fichiu@stud.uni-heidelberg.de

* Christian Homeyer
  * 3606476
  * PhD Computer Science
  * ox182@uni-heidelberg.de

* Jessica Kaechele
  * 3588787
  * MSc Applied Computer Science
  * uo251@stud.uni-heidelberg.de

* Jonas Reinwald
  * 3600238
  * MSc Applied Computer Science
  * am248@stud.uni-Heidelberg.de

## Documents

### Proposal docs
https://github.com/DonatJR/ita_ws20/tree/main/project_proposal

### Milestone document
https://github.com/DonatJR/ita_ws20/blob/main/milestone/milestone.md

## Project

### Main project goals
This repository contains scripts to download and cluster the research papers found on https://www.jmlr.org.

Our main goal is to make it easier for users to search and explore scientific papers belonging to the topic of machine learning.
For this we want to specifically arrive at a clustering (representation) that, for one, separates the different documents into correct clusters, but is also easy to work with in downstream tasks (e.g. the mentioned inclusion in some search site).

## Dataset
The dataset containing the scraped research papers is saved in the `data_2021-02-01_22-27-13.862993.json` . An example of a research paper can be seen here:
``` json
            "title": "Multiple-Instance Learning from Distributions",
            "abstract": [
                "We propose a new theoretical framework for analyzing the multiple-instance learning (MIL) setting. In MIL, training examples are provided to a learning algorithm in the form of labeled sets, or \"bags,\" of instances. Applications of MIL include 3-D quantitative structure activity relationship prediction for drug discovery and content-based image retrieval for web search. The goal of an algorithm is to learn a function that correctly labels new bags or a function that correctly labels new instances. We propose that bags should be treated as latent distributions from which samples are observed. We show that it is possible to learn accurate instance-and bag-labeling functions in this setting as well as functions that correctly rank bags or instances under weak assumptions. Additionally, our theoretical results suggest that it is possible to learn to rank efficiently using traditional, well-studied \"supervised\" learning approaches. We perform an extensive empirical evaluation that supports the theoretical predictions entailed by the new framework. The proposed theoretical framework leads to a better understanding of the relationship between the MI and standard supervised learning settings, and it provides new methods for learning from MI data that are more accurate, more efficient, and have better understood theoretical properties than existing MI-specific algorithms."
            ],
            "keywords": [
                "multiple-instance learning",
                "learning theory",
                "ranking",
                "classification"
            ],
            "author": [
                "Gary Doran"
            ],
            "ref": "https://jmlr.csail.mit.edu//papers/volume17/15-171/15-171.pdf",
            "datasource": "Journal of Machine Learning Research",
            "datasource_url": "https://jmlr.csail.mit.edu"
```

## Requirements
The requirements can be found in `requirements.txt` and can be installed with
```
pip install -r requirements.txt
```
## Usage
There are two ways to go about running the project:
* Extracting the data, creating ground truth labels and then clustering.
In this case, you must follow all three steps listed bellow.

NOTE: If you choose to follow all three steps, you should change the `input_path` parameter with the name of the newly created `json` in step one in the `yaml` configs used for step two and three.

* Clustering with the existing data
In this case, you should jump to step three.

### Step one (data extraction):
We have also included `extraction.sh`, the script responsible for extracting our data. Once run, the script will automatically start scraping all the PDFs found on https://www.jmlr.org. It will then clone the two GitHub Grobid repositories (https://github.com/kermitt2/grobid.git, and https://github.com/kermitt2/grobid_client_python) needed for setting up the Grobid server. Finally, it will set up the Grobid server, convert the scraped PDFs into XMLs and create a JSON file from the parsed XMLs.

NOTE: If the script throws an error, please make sure that Grobid is installed in a path with no parent directories containing spaces.

### Step two (create ground truth labels):
For generating ground truth labels run the file `create_gt.py` with a `--config` argument pointing to a valid yaml configuration file (`gt.yaml`).

### Step three (clustering and evaluation):
The project code can be run by executing the `main.py` script with a `--config` argument pointing to a valid yaml configuration file. If no configuration file is given, the default value `config.yaml` is used and a corresponding file has to exist in the same folder as `main.py`.

We support a variety of different options for the configuration, but encourage the usage of the provided configuration files as their configuration combinations are (well) tested.

#### Configuration
The configuration supports these options:

* __input_path__: path of the (json) data file
* __output_path__: path where all results will be saved
* __use_title__: whether to include titles of papers when loading data; possible values: True | False
* __preprocessing__:
  * __stemming__: whether to use stemming in preprocessing of data; possible values: True | False
  * __lemmatization__: whether to use lemmatization in preprocessing of data; possible values: True | False
  * __lib__: library used for preprocessing; possible values: 'spacy' | 'nltk'
  * __min_word_len__: minimum length of tokens to include in preprocessing result; possible values: Integer > 0
  * __max_word_len__: maximum length of tokens to include in preprocessing result; possible values: Integer > 0 (> min_word_len)
  * __custom_stopwords__: stopword to use in addition to standard stopwords; possible values: list of str
* __clustering__:
  * __model__: type of clustering method to use; possible values: 'KMeans' | 'Agglomerative' | 'AffinityPropagation' | 'DBSCAN' | 'MeanShift' | 'OPTICS' | 'Birch' | 'Spectral'
  * __n_clusters__: model parameter; possible values: Integer > 0; used for 'KMeans' | 'Agglomerative' | 'Birch' | 'Spectral' models
  * __agglomerative_linkage__: model parameter; possible values: 'ward' | 'complete' | 'average' | 'single'; used for 'Agglomerative' model
  * __min_samples__: model parameter; possible values: Integer > 0; used for 'DBSCAN' model
  * __eps__: model parameter; possible values: Float > 0; used for 'DBSCAN' model
  * __n_jobs__: model parameter; possible values: Integer > 0 (number of processors to use) or -1 (use all processors); used for 'DBSCAN'| 'MeanShift' | 'OPTICS' models
  * __birch_threshold__: model parameter; default value: 0.5; possible values: Integer > 1
  * __metric__: model parameter; default value: 'euclidean'; possible values: 'cosine'; used for 'DBSCAN' model
* __embedding__:
  * __dimensionality_reduction__: dimensionality reduction method to use on data before attempting to cluster; possible values: None | 'LSA' | 'SPECTRAL'
  * __n_components__: parameter for dimensionality reduction (dimension of the projected subspace); possible values: Integer > 1
* __evaluation__: whether to calculate evaluations; default value: True; possible values: True | False;  used for 'Birch' model

### Experiments
The [experiments folder](https://github.com/DonatJR/ita_ws20/tree/main/experiments) contains a variety of different notebooks used to experiment on data or evaluate methods. They are not part of the final project output, but contain the majority of the code used to get to the current project state and are therefore included in the repository. 
