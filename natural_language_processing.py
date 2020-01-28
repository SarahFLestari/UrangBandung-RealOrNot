# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the train dataset
column_names = ['text','target']
dataset = pd.read_csv('train.csv', delimiter = ',', names = column_names)


# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, len(dataset.text)):
    review = re.sub('[^a-zA-Z]', ' ', dataset.text[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)


# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 10000 )
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
classifier_Gaussian = GaussianNB()
classifier_Multinomial = MultinomialNB()
classifier_Bernoulli = BernoulliNB()

classifier_Gaussian.fit(X_train, y_train)
classifier_Multinomial.fit(X_train, y_train)
classifier_Bernoulli.fit(X_train, y_train)


# Predicting the Test set results
y_pred_gaussian = classifier_Gaussian.predict(X_test)
y_pred_multinomial = classifier_Multinomial.predict(X_test)
y_pred_bernoulli = classifier_Bernoulli.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_gaussian = confusion_matrix(y_test, y_pred_gaussian)
cm_multinomial = confusion_matrix(y_test, y_pred_multinomial)
cm_bernoulli = confusion_matrix(y_test, y_pred_bernoulli)

print("Pake 10000 Fitur")
print("Akurasi naive bayes bernoulli : ", float((cm_bernoulli[0][0]+cm_bernoulli[1][1]) / len(y_pred_bernoulli)))
print("Akurasi naive bayes multinomial : ", float((cm_multinomial[0][0]+cm_multinomial[1][1]) / len(y_pred_bernoulli)))
print("Akurasi naive bayes gaussian: ", float((cm_gaussian[0][0]+cm_gaussian[1][1]) / len(y_pred_bernoulli)))

