import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import os

data=pd.read_csv('heart.csv')
X = data.iloc[:, [0,1,2,7,12]].values
y = data.iloc[:, 13].values

# splitting data set into training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

 # Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


data.head()
data.tail()
data.describe()
data.info()
data.columns
data=data.rename(columns={'age':'Age','sex':'Sex','cp':'Cp','trestbps':'Trestbps','chol':'Chol',
'fbs':'Fbs','restecg':'Restecg','thalach':'Thalach','exang':'Exang',
'oldpeak':'Oldpeak','slope':'Slope','ca':'Ca','thal':'Thal','target':'Target'})
data.columns
data.shape
data.isnull().sum()

pd.plotting.scatter_matrix(data.loc[:,data.columns!='Target'],
                          c=['green','blue','red'],
                          figsize=[15,15],
                          diagonal='hist',
                          alpha=0.8,
                          s=200,
                          marker='*',
                          edgecolor='black')
plt.show()




