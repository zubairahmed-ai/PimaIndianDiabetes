import numpy as np  # linear algebra
from sklearn import cross_validation, metrics  # Additional scklearn functions
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from matplotlib.pylab import rcParams
from sklearn.pipeline import Pipeline

current_file = os.path.abspath(os.path.dirname(__file__))
csv_filename = os.path.join(current_file, 'data/diabetes.csv')
train2 = pd.read_csv(csv_filename)

target = 'Outcome'

rcParams['figure.figsize'] = 12, 4

drop_elements = ['Outcome']

train2.loc[train2['Insulin'] <= 211.5, 'Insulin'] = 0
train2.loc[(train2['Insulin'] > 211.6) & (train2['Insulin'] <= 423), 'Insulin'] = 1
train2.loc[(train2['Insulin'] > 424) & (train2['Insulin'] <= 634.5), 'Insulin'] = 2
train2.loc[train2['Insulin'] > 634.5, 'Insulin'] = 3
#
train2.loc[train2['DiabetesPedigreeFunction'] <= 0.663, 'DiabetesPedigreeFunction'] = 0
train2.loc[(train2['DiabetesPedigreeFunction'] >= 0.664) & (train2['DiabetesPedigreeFunction'] <= 1.249), 'DiabetesPedigreeFunction'] = 1
train2.loc[(train2['DiabetesPedigreeFunction'] >= 1.250) & (train2['DiabetesPedigreeFunction'] <= 1.835), 'DiabetesPedigreeFunction'] = 2
train2.loc[(train2['DiabetesPedigreeFunction'] >= 1.836), 'DiabetesPedigreeFunction'] = 3

y = train2['Outcome'].values
train = train2.drop(drop_elements, axis=1)

X = train.iloc[:, :].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.fit_transform(X_test)

pca = PCA(n_components=7)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

lr = LogisticRegression()

pipe = Pipeline([('pca', pca), ('logistic', lr)])
pipe.fit(X_train_pca, y_train)
predictions = pipe.predict(X_test_pca)
acc = accuracy_score(y_test, predictions)

print acc * 100

forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train_pca, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
labels = train.columns

fpred = forest.predict(X_test_pca)

print ("Accuracy : %.4g" % metrics.accuracy_score(fpred, y_test) * 100)

