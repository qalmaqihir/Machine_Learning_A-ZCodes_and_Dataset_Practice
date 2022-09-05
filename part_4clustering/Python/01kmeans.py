# K-Means Clustering

# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the Dataset
dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv"
dataset=pd.read_csv(dataset_path)
# print(dataset)
X = dataset.iloc[:,[3,4]].values
# print(X)

# Using the elbow method to find the optimal number of clusters (k)
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    k_means=KMeans(n_clusters=i,init='k-means++',random_state=42)
    k_means.fit(X)
    wcss.append(k_means.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel("Number of clusters")
plt.ylabel("WCSS value")
plt.show()


# Training the K-Means model on the dataset
k_means=KMeans(n_clusters=5, init='k-means++')
clusters = k_means.fit_predict(X)
print(clusters)
# Visualizing the clusters
plt.scatter(X[clusters==0,0],X[clusters==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[clusters==1,0],X[clusters==1,1],s=100,c='blue',label='Cluster 2')
plt.scatter(X[clusters==2,0],X[clusters==2,1],s=100,c='green',label='Cluster 3')
plt.scatter(X[clusters==3,0],X[clusters==3,1],s=100,c='yellow',label='Cluster 4')
plt.scatter(X[clusters==4,0],X[clusters==4,1],s=100,c='magenta',label='Cluster 5')
plt.scatter(k_means.cluster_centers_[:,0],k_means.cluster_centers_[:,1],s=300,c='cyan',label='Centroids')
plt.title('Clusters of Customers')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()

