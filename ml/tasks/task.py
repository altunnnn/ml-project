import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

print(df.head())

vectorizer = TfidfVectorizer(max_features=1000) 
X = vectorizer.fit_transform(df['text'])

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

sse = kmeans.inertia_
print(f'Sum of Squared Errors (SSE): {sse}')

silhouette_avg = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')
