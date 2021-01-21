Evaluation pipeline
---
# Problem
- We want to create supervision for clustering research papers
- We currently have only papers from 1 superclass, i.e. "machine learning"
- We have access to key words for each paper:
    - these are 4 to 7 in total, but can consist of multiple words
    - this means we actually have 10 words max per paper
How to get supervising labels for this kind of data without manually labeling?

How can you objectively determine the overall number of clusters?

# Naive approach
- Create super set of all key words
- Create binary vector for each paper, where a single 1 means the paper has this key word as label
- Compute Hamming distance between papers
- Use distance matrix to create clusters of papers
- Use these new labels as ground truth supervision for clustering from text corpus

# Better
- Query all key words with text model
- Tokenize key words to create super class, e.g. several key words can be combined
under the umbrella "optimization".
- Greedily create clusters for papers that share a common umbrella
- Stop if cluster contains only single paper
- Go one hierarchy higher or manually assign clusters

# Manual 
- Read papers and corpus
- Assign manual labels as to what papers share common topic
- Papers will have overlap which will make this hard
- This will be subjective as some papers have no common label but we may want to label them not singularly
- Some papers will have overlap but we want them to be seperated
- will take quite some time


# Hybrid Solution
- Use on of the automatic techniques from above and manually check for correctness
- Make case decisions for individual "problematic" papers
