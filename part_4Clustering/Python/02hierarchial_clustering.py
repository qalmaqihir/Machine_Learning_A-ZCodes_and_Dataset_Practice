# Hierarchial Clustring


# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataset_path="../../DatasetsMall_Customers.csv"
dataset=pd.read_csv(dataset_path)
# print(dataset)
X = dataset.iloc[:,[3,4]].values
# print(X)

# Using Dendrograms to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Observation points(Customers)')
plt.ylabel("Euclidean Distances")
plt.show()

# Training the Hierarchical clustering
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
clusters=hc.fit_predict(X)
print(clusters)

# Visualizing the clusters
plt.scatter(X[clusters==0,0],X[clusters==0,1],s=100,c='red',label='c 1')
plt.scatter(X[clusters==1,0],X[clusters==1,1],s=100,c='blue',label='c 2')
plt.scatter(X[clusters==2,0],X[clusters==2,1],s=100,c='green',label='c 3')
plt.scatter(X[clusters==3,0],X[clusters==3,1],s=100,c='yellow',label='c 4')
plt.scatter(X[clusters==4,0],X[clusters==4,1],s=100,c='magenta',label='c 5')
plt.legend()
plt.colorbar()
plt.clim(0,5)
plt.title('Clusters of Customers')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(loc='upper left', frameon=True, ncol=3)
plt.show()

