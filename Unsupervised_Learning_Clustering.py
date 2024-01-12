# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:48:19 2023

@author: harsh
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist

# Load the dataset
data = pd.read_csv('Wisconsin_breast_cancer.csv')

# Select relevant features for clustering
features = data.drop(['id', 'diagnosis'], axis=1)

# Perform hierarchical agglomerative clustering
model = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster_labels = model.fit_predict(features)

# Create linkage matrix
Z = linkage(features, 'ward')

# Create dendrogram
dendrogram(Z, leaf_rotation=80, leaf_font_size=9, labels=data['diagnosis'].values)
plt.title('Hierarchical Agglomerative Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

# Evaluate the model using the cophenetic correlation coefficient
c, coph_dists = cophenet(Z, pdist(features))
print(f'Cophenetic Correlation Coefficient: {c}')

"""
The cophenetic correlation coefficient of 0.785 indicates a moderate to strong correlation 
between the original distances of the samples and the dendrogrammatic distances at which samples
 were merged. This suggests that the dendrogram preserves the pairwise distances between the 
 original unclustered data points fairly well. 
 In clustering, a higher cophenetic correlation coefficient indicates that the clustering 
 algorithm has done a good job of preserving the original distances, with 1 being perfect 
 preservation.

Given the output cophenetic correlation coefficient value of approximately 0.785, we can infer 
that the hierarchical clustering model has a reasonably good fit to the data.


"""