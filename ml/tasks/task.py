import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Loading the dataset
df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

# Displaying the first few rows of the dataset to understand its structure
print(df.head())

# Data Preprocessing
# Initializing the TfidfVectorizer with max_features set to 1000 to reduce dimensionality
vectorizer = TfidfVectorizer(max_features=1000) 
X = vectorizer.fit_transform(df['text'])

# K-means Clustering
# Creating KMeans instance with 3 clusters and a fixed random state for reproducibility
kmeans = KMeans(n_clusters=3, random_state=42)
# Fit the model to our data
kmeans.fit(X)

# Calculating the Sum of Squared Errors (SSE), which measures the variance of the data within each cluster
sse = kmeans.inertia_
print(f'Sum of Squared Errors (SSE): {sse}')

# Calculating the Silhouette Score, which measures how similar an object is to its own cluster compared to other clusters
silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')
