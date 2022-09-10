# Natural Language Processing

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset, its a tsv file and we will ignor the " to avoid processing errors

dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python/Restaurant_Reviews.tsv"
dataset=pd.read_csv(dataset_path,delimiter='\t', quoting=3)
print(dataset.head())


# Cleaning the texts
import re
import nltk

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus =[] # will contained all our cleaned reviews
for i in range(0,1000):
    review=re.sub("[^a-zA-Z]"," ", dataset['Review'][i])
    review=review.lower()
    review=review.split()

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review=[ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)

    corpus.append(review)

print(corpus)

# Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer

# cv = CountVectorizer(max_features=1500)
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
# print(len(X))
y=dataset.iloc[:,-1].values
# print(len(y))
# print(len(X[0]))
#
# Splitting the dataset into the Training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.25)

# Training the Naive Bayes model training set
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, GaussianNB
# binary_nb = BernoulliNB()
# binary_nb.fit(x_train, y_train)
gaussain_nb = GaussianNB()
gaussain_nb.fit(x_train, y_train)

# Predicting the test results
y_pred = gaussain_nb.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm=confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))

