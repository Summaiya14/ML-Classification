import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("name_gender_dataset.csv")

X = dataset.iloc[0:2000, [0,2,3]].values
y = dataset.iloc[0:2000, 1].values

print(dataset.isna().sum())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X[:,0]=le.fit_transform(X[:,0].astype(str))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#Feature Scaling (all the values of the features in the same range)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc.transform(X_test[:, 1:]) 

#training the K-NN model on the training set
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5 ,metric = 'minkowski', p = 2)
classifier_knn.fit(X_train, y_train)

#training the logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

#training the naive bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)

#training the decision tree classification model on the training set
from sklearn.tree import DecisionTreeClassifier
classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_dt.fit(X_train, y_train)


#training the random forest classification model on the training set
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)


#Cross Validation Leave one out technique
# loocv to automatically evaluate the performance of a KNN classifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
cv = LeaveOneOut()
scores_knn = cross_val_score(classifier_knn, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy of KNN classifier: %.3f (%.3f)' % (mean(scores_knn), std(scores_knn)))
# loocv to automatically evaluate the performance of a logistic regression classifier
scores_lr = cross_val_score(classifier_lr, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy of logistic regression: %.3f (%.3f)' % (mean(scores_lr), std(scores_lr)))
# loocv to automatically evaluate the performance of a Naive Bayes classifier
scores_nb = cross_val_score(classifier_nb, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy of Naive Bayes classifier: %.3f (%.3f)' % (mean(scores_nb), std(scores_nb)))
# loocv to automatically evaluate the performance of a Decision tree classifier
scores_dt = cross_val_score(classifier_dt, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy of Decsion tree classifier: %.3f (%.3f)' % (mean(scores_dt), std(scores_dt)))
# loocv to automatically evaluate the performance of a Random forest classifier
scores_rf = cross_val_score(classifier_rf, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)
print('Accuracy of Random forest classifier: %.3f (%.3f)' % (mean(scores_rf), std(scores_rf)))