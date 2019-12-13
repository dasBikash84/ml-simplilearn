#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 01:07:33 2019

@author: bikash
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs

from sklearn.cluster import KMeans

from sklearn.datasets import load_sample_image
#china = load_sample_image('flower.jpg')
china = load_sample_image('china.jpg')
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)

data =china/255.0
data = data.reshape(china.shape[0]*china.shape[1],china.shape[2])
data.shape

plot_pixels(data,title='Plot all colors')

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(16)
kmeans.fit(data)
new_colors = kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixels(data,colors=new_colors,title='Reduced in 16 colors')

china_recolored = new_colors.reshape(china.shape)
fig,ax=plt.subplots(1,2,figsize=(16,6),
                    subplot_kw=dict(xticks=[],yticks=[]))

fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image',size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-Color Image',size=16)




def plot_pixels(data,title,colors=None,N=10000):
    if colors is None:
        colors = data
    
    #random sub-set
    rng=np.random.RandomState(0)
    i=rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R,G,B = data[i].T
    
    fig,ax=plt.subplots(1,2,figsize=(16,6))
    ax[0].scatter(R,G,color=colors,marker='.')
    ax[0].set(xlabel='Red',ylabel='Green',xlim=(0,1),ylim=(0,1))
    
    
    ax[1].scatter(R,B,color=colors,marker='.')
    ax[1].set(xlabel='Red',ylabel='Blue',xlim=(0,1),ylim=(0,1))
    
    fig.suptitle(title,size=20)
    
    
    
    
    
    
    
    
    
    
    
    