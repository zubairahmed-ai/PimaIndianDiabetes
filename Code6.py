from sklearn.svm import SVC
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from matplotlib.pylab import rcParams

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

svm = SVC(kernel='linear', C=0.2, random_state=0)
svm.fit(X_train_pca, y_train)
spred = svm.predict(X_test_pca)
print "Accuracy with SVM {0}".format(accuracy_score(spred, y_test) * 100)

