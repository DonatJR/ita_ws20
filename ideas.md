# How do we get this evaluation done?!

# Goal
Compute similarity between papers based on ground truth key words!
--> How to compute distance between set of key words?

# W2Vec
Assume we use w2vec on key words (not a single word):
- For each kword:
    - vec = w2vec ( kword)
- this means for each paper:
    - feature = [vec1, vec2, ..., vecN]
How to compute distance between features now?
W2Vec is great for computing distances due to cosine similaritiy, but what if we have a vector of 
w2vec's?

# Previously
For each kword in set(all_kwords):
    - Mark index of kword in set(all_kwords) as 1 if paper has it
    - All other indices are 0

--> For each paper, we get a sparse binary vector

# Example of W2Vec
- Multiple key words that are kinda similar, e.g. "multiple-instance learning" and "deep-learning" get mapped to nearly same vector --> Unify these
- WHAT IF we bin the w2vec to similar vectors and keep these?

- Is w2vec surjective?!?! Can we go back to word?
        - vect1, vect2, vect3 --> word

# Problem right now
Our dataset has a lot of single standalone key words 
--> binary vectors get very sparse and the few features we have get meaningless


Def. Feature vector:
- [feat1, feat2, feat3, ..., featN]  --> This could a be tensor
- Each feat can be a value describing the key words
- E.g. curently feat = [0, 1] for whether key word is in paper or not
- but this could also be from w2vec \in \real.
