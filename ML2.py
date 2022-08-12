from importlib.resources import contents
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import scipy as sp
import sys

# Exercise taken from the book Building Machine Learning Systems with Python Richert Coelho
def dist_raw(v1, v2):
    v1_normalized = v1/sp.linalg.norm(v1.toarray())
    v2_normalized = v2/sp.linalg.norm(v2.toarray())
    delta = v1_normalized - v2_normalized
    return sp.linalg.norm(delta.toarray())

vectorizer = CountVectorizer(min_df=1,stop_words='english' )
posts = ["This is a toy post about machine learning. Actually, it contains not much interesting stuff.", "Imaging databases provide storade capabilities.", "Most imaging databases safe images permanently.", "Imaging databases store data.", "Imaging databases store images. Imaging databases store images. Imaging databases store images."]

X_train = vectorizer.fit_transform(posts)

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
num_samples, num_features = X_train.shape

best_doc = None
best_dist = sys.maxsize
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if post==new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist_raw(post_vec, new_post_vec)
    print ("=== Post %i with dist=%.2f: %s"%(i, d, post))
    if d<best_dist:
        best_dist = d
        best_i = i
print("Best post is %i with dist=%.2f"%(best_i, best_dist))
