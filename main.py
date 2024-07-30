import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#reading in data
df = pd.read_csv('/Users/rushilgangisetty/Downloads/news.csv')

#printing shape and head of data
print(df.shape)
print(df.head())

#gettings labels for data
labels = df.label
print(labels.head())

#splitting data into training and validation sets

X_train, X_validation, y_train, y_validation, = train_test_split(df['text'], labels, test_size = .2, random_state = 7)

#initializing a TfidVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = .7)

#fitting and transforming the test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfid_validation = tfidf_vectorizer.transform(X_validation)

#initializing passive aggressive classifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

#predicting on test set and scoring model accuracy
y_pred = pac.predict(tfid_validation)
score = accuracy_score(y_validation, y_pred)
print(f'Accuracy: {round(score * 100,2)}%')

#getting confusion matrix to see num of false negatives/positves etc
print(confusion_matrix(y_validation, y_pred, labels = ['FAKE', 'REAL']))