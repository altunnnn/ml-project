import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')

# Update with the path to your dataset
df = pd.read_csv('Apple-Twitter-Sentiment-DFE.csv', encoding='latin1')

# Display the first few rows of the dataset
print(df.head())

# Defining a function to include converting to lowercase, removing URLs, non-alphabetic characters, hashtags, and stopwords
def preprocess_text(text):
    text = text.lower() if isinstance(text, str) else ""

    text = re.sub(r'http\S+', '', text)  # Removes http links
    text = re.sub(r'www\S+', '', text)  # Removes www links

    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)

    # Remove common English stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)

    return text

# Check if the 'text' column exists in the DataFrame
if 'text' in df.columns:
    # Apply the preprocessing function to the 'text' column and store the result in a new column 'clean_text'
    df['clean_text'] = df['text'].apply(preprocess_text)
else:
    print("Error: 'text' column not found in the DataFrame.")

if 'clean_text' in df.columns:
    # Vectorize the cleaned text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['clean_text'])
    
    # Separate the features (X) and target variable (y)
    y = df['sentiment']
    
    #Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train a logistic regression model on the training data
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Model Evalution | Predicting
    y_pred = model.predict(X_test)

    # Calculating and printing the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    # Classification report
    classif_report = classification_report(y_test, y_pred)
    print("Classification_report :")
    print(classif_report)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion_matrix :")
    print(conf_matrix)
else:
    print("Error: 'clean_text' column not found in the DataFrame.")
