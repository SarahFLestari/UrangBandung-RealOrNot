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

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
import xgboost as xgb
classifier_multinomial = MultinomialNB()
classifier_knn = KNeighborsClassifier(n_neighbors = 11, metric = 'minkowski', p = 2)
classifier_logr = LogisticRegression(random_state = 0)
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_xg=xgb.XGBClassifier(random_state=1,learning_rate=0.15)

classifier_logr.fit(X_train, y_train)
classifier_knn.fit(X_train, y_train)
classifier_dt.fit(X_train, y_train)
classifier_rf.fit(X_train, y_train)
classifier_multinomial.fit(X_train, y_train)
classifier_xg.fit(X_train,y_train)


# Predicting the Test set results

y_pred_logr = classifier_logr.predict(X_test)
y_pred_knn = classifier_knn.predict(X_test)
y_pred_dt = classifier_dt.predict(X_test)
y_pred_rf = classifier_rf.predict(X_test)
y_pred_multinomial = classifier_multinomial.predict(X_test)
y_pred_xgboost = classifier_xg.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_logr = confusion_matrix(y_test, y_pred_logr)
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_dt = confusion_matrix(y_test, y_pred_dt)
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_multinomal = confusion_matrix(y_test,y_pred_multinomial)
cm_xg = confusion_matrix(y_test,y_pred_xgboost)

print("Pake 10000 Fitur")
print("Akurasi logistic regression : ", float((cm_logr[0][0]+cm_logr[1][1]) / len(y_pred_logr)))
print("Akurasi KNN : ", float((cm_knn[0][0]+cm_knn[1][1]) / len(y_pred_knn)))
print("Akurasi decision tree : ", float((cm_dt[0][0]+cm_dt[1][1]) / len(y_pred_dt)))
print("Akurasi random forest : ", float((cm_rf[0][0]+cm_rf[1][1]) / len(y_pred_rf)))
print("Akurasi multinomial : ", float((cm_multinomal[0][0]+cm_multinomal[1][1]) / len(y_pred_multinomial)))
print("Akurasi Xg boost : ", float((cm_xg[0][0]+cm_xg[1][1]) / len(y_pred_xgboost)))

y_pred_el = []
for i in range(0,len(y_pred_logr)):
    count_y,count_n = 0,0
    if (y_pred_logr[i] == '0'):
        count_n += 1
    elif (y_pred_logr[i] == '1') :
        count_y += 1
        
    if (y_pred_rf[i] == '0'):
        count_n += 1
    elif (y_pred_rf[i] == '1') :
        count_y += 1
        
    if (y_pred_multinomial[i] == '0'):
        count_n += 1
    elif (y_pred_multinomial[i] == '1') :
        count_y += 1
        
    if (count_y<=count_n):
        y_pred_el.append(0)
    elif (count_y>count_n) :
        y_pred_el.append(1)


accuracy = 0
for j in range(0,len(y_pred_el)):
    if (y_pred_el[j] == int(y_test[j])):
        accuracy += 1
        
print("Akurasi Ensemble learning : ", float( accuracy / len(y_pred_el)))


        
    
        
    
