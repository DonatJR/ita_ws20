# Project Title:

## General

### Team members

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


### Existing Code Fragments
In case you are choosing a project for which there exists code that you are using, you must clearly indicate those parts of your project and link to them.
TODO: we are not using any existing code, right? --> replace with "No existing code fragments used." 

### Utilized libraries
TODO: Provide a list of all required libraries to successfully run your system. Ideally, this comes in the __form of a separate requirements.txt file__.

TODO: Should we list this here or does the requirements file speak for itself?

### Contributions
We had some issues with non-matching email addresses in commits. This results in some team members not showing up as project contributors (or with reduced contributions), as their GitHub profile cannot be linked to the git user used to make the commits. The commit history nevertheless reflects all contributions correctly, just without a direct link to the respective GitHub profile.

## Project State

### Planning State
TODO: brief overview of everything we have done until now

### Future Planing
TODO: brief timeline of tasks to be done in second phase of project

### High-level Architecture Description
TODO: high level description of _project structure_ and _pipeline_.

### Experiments
TODO: In case that you already have some results from initial experiments, you may detail the results and implications. We strongly encourage you to already provide simple baselines.

We decided to perform initial clustering experiments on the dataset we gathered in order to sketch out the overall algorithmic pipeline. For these experiments we performed the following steps:
1. Preprocessing to exctract informative tokens
2. Vectorization into TFIDF and bag of words
3. Clustering using algorithms from ```gensim``` and ```sklearn```
4. Visualization with projected data.

## Data Analysis

### Data Sources
We mentioned two data sources for research papers in our project proposal:

- [jmlr](https://jmlr.csail.mit.edu) - an international forum for the electronic and paper publication of scholarly articles in all areas of machine learning

- [ann](http://aan.how) - a website maintained by Yale University's Learning and Information Group that provides a corpus consisting of over 400 scientific papers on NLP in plain text.

We ended up using only the former source.

[ann](http://aan.how) proved an unreliable data source because: 

- we downloaded about 200 papers in txt format from [ann](http://aan.how) and randomly examinted 10 of them: they all contained spelling/formating mistakes, i.e. *Abstra-           ct* as a paragraph title
- the quality of the files made it hard to find regex for information extraction
- even though [ann](http://aan.how) has to each paper a html page where the abstract could be found, after examination, we were once again facing the problem of an error-riden abstract

[jmlr](https://jmlr.csail.mit.edu) proved an excellent data source because: 

- the research papers are ordered in volumes. There are currenlty 21 volumes available. Each volume has its own website, i.e. https://jmlr.csail.mit.edu/papers/v18/
- the volumes' URL differ only by the number of the volume
- to each research paper there are three links, two of them being:
   - a html page containing the abstract, names of the authors, title of the paper
   - the URL to the pdf
- almost all the pdf's have the same structure: 

Abstact

<abstract_text>

Keywords: <list_of_keywords>

- having a rigid strcuture allowed us to use regex to extract information

We decided to use the following information from a research paper: 
- title
- authors
- ref
- data source url
- pdf url
- keywords. 

Based on this, we scraped the html page of the research paper and converted the pdf into a txt file to extract the keywords using regex. The extracted information was saved in a json.
   

### Preprocessing
TODO: Detail any preprocessing steps you have taken to ensure proper data quality. This can include unicode normalization, length normalization, text sanitizing, etc

We do standard preprocessing before clustering. We tried out the three libarries: ```gensim```, ```spacy``` 
and ```nltk```. After dealing with some problems regarding lemmatization in ```gensim``` (and the depending 
```pattern```, we found out in an [issue](https://github.com/RaRe-Technologies/gensim/issues/2716) on their official github repository, 
that the library was not intended for preprocessing. We therefore dropped this one.  
Our preprocessing consists of:
1. Removal non-alphabetic characters
2. Tokenization
3. Lemmatization
4. Stemming
5. Length normalization
6. Stop word removal
We use the respective standard models for the english language of the libaries.

The pipeline is in part configurable, because we want to compare our results towards raw text inputs. 
After taking a first look at the resulting tokens for our first data, we realized the following:
- Most papers share very common words for their common denominator field. 
- Since we tried this out on  machine learning papers, all tokenized abstracts include e.g."_data_" with a high frequency. 
- It should be benificial to include such common words and buzzwords in the stop word list

Currently our clustering performance will be suboptimal since we might not have very informative tokens for the respective topics.

### Basic Statistics
No. of samples: 1261

#### Keywords Statistics:
A keyword can contain mutiple words, i.e, *bayesian statistics*.

No of papers without keywords: 153

Reason: no keywords present in the papers

![](images/keywords.png)

![](images/top_keywords.png)

996 papers have no keywords in top 10 

249 papers have 1 keyword in top 10

16 papers have 2 keywords in top 10 

The maximum no of top 10 keywords contained by a single papers is 2

![r](images/distribution_top_keywords.png)

![](images/top_keywords_wordcloud.png)

#### Abstract Statistics:
No of papers without abstract: 36

Reason: we have two sources for the abstract: 
 
i) a html page where the abstract is in a meta tag. This abstract contains inline math latex notation.
  
ii) the pdf where the abstract doesn't contain any latex notation

We extract them both (with the intetion to later compare the quality), but only use the one from the html page in our data files.
  
After investigating why 36 papers seemingly have no abstract, it seems that for them the abstract is under a different html tag. We intend to correct this in the 
scraping script, but we found the observation interesting, because it shows that scraping is very specific to the website we intend to extract information from.

![](images/abstract_words.png)


### Examples
 
      "title": "Online Sufficient Dimension Reduction Through Sliced Inverse Regression",
      "abstract": "Sliced inverse regression is an effective paradigm that achieves the goal of dimension reduction through replacing high dimensional covariates with a small number of linear combinations. It does not impose parametric assumptions on the dependence structure. More importantly, such a reduction of dimension is sufficient in that it does not cause loss of information. In this paper, we adapt the stationary sliced inverse regression to cope with the rapidly changing environments. We propose to implement sliced inverse regression in an online fashion. This online learner consists of two steps. In the first step we construct an online estimate for the kernel matrix; in the second step we propose two online algorithms, one is motivated by the perturbation method and the other is originated from the gradient descent optimization, to perform online singular value decomposition. The theoretical properties of this online learner are established. We demonstrate the numerical performance of this online learner through simulations and real world applications. All numerical studies confirm that this online learner performs as well as the batch learner.",
      "keywords": [
        "Dimension reduction",
        "online learning",
        "perturbation",
        "singular value decomposition",
        "sliced inverse regression",
        "gradient descent"
      ],
      "author": [
        "Zhanrui Cai",
        "Runze Li",
        "Liping Zhu"
      ],
      "ref": "https://jmlr.csail.mit.edu/papers/volume21/18-567/18-567.pdf",
      "datasource": "Journal of Machine Learning Research",
      "datasource_url": "https://jmlr.csail.mit.edu/"
    

#### Data - Reflection
While doing the first assignment we kept asking ourselves why: why should we spend so much time looking for a conversion library or come up with regex expressions.
We take it all back: *now* we understand why. By working with regex in the first assignment, it seemed natural to use this newly acquired knowledge for information extraction. Furthemore, after having tested out so many conversion libraries, we already knew which ones we could trust.

We also believe in the power of recycling - this is where the second assignment comes into play. For the worldcloud of keywords and the top 10 keywords plot, we slightly modify our code from the simpsons' assignment.
