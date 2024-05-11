import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

print(df.head())

# Data Preprocessing
# Initializing the TfidfVectorizer with max_features set to 1000 for reducing dimensionality
vectorizer = TfidfVectorizer(max_features=1000)

# Transform the 'text' column into a matrix of TF-IDF features
# TF-IDF - Term frequency–inverse document frequency, is a measure of importance of a word to a document in a collection or corpus, adjusted for the fact that some words appear more frequently in general
X = vectorizer.fit_transform(df['text'])

# K-means Clustering
# Initializing KMeans with n_clusters=3 and a fixed random state for reproducibility
kmeans = KMeans(n_clusters=3, random_state=42)

# Fittinig the model to the data
kmeans.fit(X)

# Evaluation Metrics
# Sum of Squared Errors (SSE)
sse = kmeans.inertia_
print(f'Sum of Squared Errors (SSE): {sse}')

# Silhouette Score
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

# Note that Cohesion and Separation are not directly calculated by scikit-learn's metrics.
# However, they can be inferred from the SSE and Silhouette Score:
# - Lower SSE indicates better cohesion within clusters.
# - Higher Silhouette Score indicates better separation between clusters.

# SSE is the sum of the squared differences between each observation and its group's mean. 
# Formula: https://hlab.stanford.edu/brian/error_1.gif

# The Silhouette Coefficient is calculated using the mean intra-cluster distance ( a ) and the mean nearest-cluster distance ( b ) for each sample. 
# The Silhouette Coefficient for a sample is (b - a) / max(a, b) .

# Cohesion
#   Cohesion refers to how closely the data points within a cluster are related. 
#   A lower SSE indicates better cohesion because it means the sum of the squared distances from each point to its assigned center is minimized, 
#   suggesting that points within the same cluster are closer to each other. 
# In this context a lower SSE would suggest that the tweets within each identified cluster are more similar to each other in terms of their content or sentiment, indicating strong cohesion within those clusters.

# Separation
#   Separation refers to how distinct the clusters are from each other. 
#   A higher Silhouette Score indicates better separation because it means the average distance between points in the same cluster (a) is greater than the average distance between points in different clusters (b), 
#   where the Silhouette Score is defined as (b - a) / max(a, b). 


