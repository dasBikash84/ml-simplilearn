#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 00:24:25 2019

@author: bikash
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

X,y_true = make_blobs(n_samples=300,cluster_std=.9,centers=4,random_state=0)
plt.scatter(X[:,0],X[:,1],s=50)


kmeans = KMeans(n_clusters=6)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:,0],X[:,1],s=50,c=y_kmeans)

ks = kmeans.cluster_centers_[kmeans.predict(X)]
