import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Data preprocessing
data = pd.read_csv("name_gender_dataset.csv")
print(data.describe())
print(data.info())
data["Gender"].describe()
data["Gender"].unique()
#First, analysing the target variable
a = data["Gender"]
sns.countplot(a)
plt.show()

#Analysing the 'Count' feature
sns.barplot(data["Count"],a)
plt.show()

#Analysing the 'Probability' feature
sns.barplot(data["Probability"],a)
plt.show()

# #separate dependent and independent variables in the dataset
# X = data.iloc[:,[0,2,3]].values
# y = data.iloc[:,1].values

# #checking null values
# print(data.isna().sum())
# #No null values

# #Label Encoding
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# X[:, 0] = le.fit_transform(X[:, 0].astype(str))

# #splitting testing data and training data from dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# #Feature Scaling (all the values of the features in the same range)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train[:, 1:] = sc.fit_transform(X_train[:, 1:])
# X_test[:, 1:] = sc.transform(X_test[:, 1:])

# #training the K-NN model on the training set
# from sklearn.neighbors import KNeighborsClassifier
# classifier_knn = KNeighborsClassifier(n_neighbors = 5 ,metric = 'minkowski', p = 2)
# classifier_knn.fit(X_train, y_train)

# #training the logistic regression model on the training set
# from sklearn.linear_model import LogisticRegression
# classifier_lr = LogisticRegression(random_state = 0)
# classifier_lr.fit(X_train, y_train)

# #training the naive bayes model on the training set
# from sklearn.naive_bayes import GaussianNB
# classifier_nb = GaussianNB()
# classifier_nb.fit(X_train, y_train)

# #training the decision tree classification model on the training set
# from sklearn.tree import DecisionTreeClassifier
# classifier_dt = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier_dt.fit(X_train, y_train)

# #training the random forest classification model on the training set
# from sklearn.ensemble import RandomForestClassifier
# classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier_rf.fit(X_train, y_train)

# #Confusion Matrix and accuracy score
# from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# from sklearn import metrics
# y_pred_knn = classifier_knn.predict(X_test)
# cm_knn = confusion_matrix(y_test, y_pred_knn)
# print(cm_knn)
# print("The accuracy of KNN is "+str(metrics.accuracy_score(y_test,y_pred_knn)*100)+"%")
# y_pred_lr = classifier_lr.predict(X_test)
# cm_lr = confusion_matrix(y_test, y_pred_lr)
# print(cm_lr)
# print("The accuracy of Logistic Regression is "+str(metrics.accuracy_score(y_test,y_pred_lr)*100)+"%")
# y_pred_nb = classifier_nb.predict(X_test)
# cm_nb = confusion_matrix(y_test, y_pred_nb)
# print(cm_nb)
# print("The accuracy of Naive Bayes Classifier is "+str(metrics.accuracy_score(y_test,y_pred_nb)*100)+"%")
# y_pred_dt = classifier_dt.predict(X_test)
# cm_dt = confusion_matrix(y_test, y_pred_dt)
# print(cm_dt)
# print("The accuracy of Decision Tree Classifier is "+str(metrics.accuracy_score(y_test,y_pred_dt)*100)+"%")
# y_pred_rf = classifier_rf.predict(X_test)
# cm_rf = confusion_matrix(y_test, y_pred_rf)
# print(cm_rf)
# print("The accuracy of Random Forest Classification is "+str(metrics.accuracy_score(y_test,y_pred_rf)*100)+"%")

# #ROC curve and classification report
# target_names = ['class 0', 'class 1']
# print("Classification report for KNN:\n", classification_report(y_test, y_pred_knn, target_names=target_names))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_knn,pos_label=2)
# metrics.plot_roc_curve(classifier_knn, X_test, y_test)
# plt.show()
# print("Classification report for Logistic Regresssion:\n", classification_report(y_test, y_pred_lr, target_names=target_names))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_lr,pos_label=2)
# metrics.plot_roc_curve(classifier_lr, X_test, y_test)
# plt.show()
# print("Classification report for Naive Bayes Classiifier:\n", classification_report(y_test, y_pred_nb, target_names=target_names))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_nb,pos_label=2)
# metrics.plot_roc_curve(classifier_nb, X_test, y_test)
# plt.show()
# print("Classification report for Decision Tree Classifier:\n", classification_report(y_test, y_pred_dt, target_names=target_names))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_dt,pos_label=2)
# metrics.plot_roc_curve(classifier_dt, X_test, y_test)
# plt.show()
# print("Classification report for Random Forest Classification:\n", classification_report(y_test, y_pred_rf, target_names=target_names))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_rf,pos_label=2)
# metrics.plot_roc_curve(classifier_rf, X_test, y_test)
# plt.show()

# #Error calculation
# from sklearn.metrics import mean_absolute_error
# print("Mean absolute error for knn classifier:", mean_absolute_error(y_test, y_pred_knn))
# print("Mean absolute error for logistic regression:", mean_absolute_error(y_test, y_pred_lr))
# print("Mean absolute error for naive bayes classifier:", mean_absolute_error(y_test, y_pred_nb))
# print("Mean absolute error for decision tree classifier:", mean_absolute_error(y_test, y_pred_dt))
# print("Mean absolute error for random forest classifier:", mean_absolute_error(y_test, y_pred_rf))




