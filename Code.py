import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
import  os

# script.py
current_file = os.path.abspath(os.path.dirname(__file__))
csv_filename = os.path.join(current_file, 'data/pimaindiandiabetes.csv')
df = pd.read_csv(csv_filename)

# print len(df[df['Pregnancies'].isnull()])
# print len(df[df['Glucose'].isnull()])
# print len(df[df['BloodPressure'].isnull()])
# print len(df[df['SkinThickness'].isnull()])
# print len(df[df['Insulin'].isnull()])
# print len(df[df['BMI'].isnull()])
# print len(df[df['DiabetesPedigreeFunction'].isnull()])
# print len(df[df['Age'] == 0])
# print len(df[df['Outcome'].isnull()])


import matplotlib.pyplot as plt
import seaborn as sns
# sns.set(style='whitegrid', context='notebook')

cols=['Pregnancies', 'Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']
# sns.pairplot(df[cols], size=2.5)
# plt.show()

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1)
# hm = sns.heatmap(cm, cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size': 15},yticklabels=cols,xticklabels=cols)
# plt.show()


X = df.iloc[:, :-1].values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)

# model = LogisticRegression()
# rfe = RFE(model, 3)
# rfe = rfe.fit(X_train_std, y_train)
# print(rfe.support_)
# print(rfe.ranking_)


model = ExtraTreesClassifier()
model.fit(X_train_std, y_train)
# display the relative importance of each attribute

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train_std.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# summarize the selection of the attributes

# print df.info()

#
df.loc[df['Glucose'] <=110, 'Glucose'] = 0
df.loc[df['Glucose'] >110, 'Glucose'] = 1
# df['GlucoseLevel'] = pd.qcut(df['Glucose'], 3)
# print (df[['GlucoseLevel', 'Outcome']].groupby(['GlucoseLevel'], as_index=False).sum())

print df[['Glucose', 'Outcome']].groupby(['Outcome'], as_index=False).mean()

# print '************************'
#
# print(pd.crosstab(df['Outcome'], df['Glucose']))
#
# print '++++++++++++++++++++++++++++'

print df[['BMI', 'Outcome']].groupby(['Outcome'], as_index=False).mean()
print df[['Age', 'Outcome']].groupby(['Outcome'], as_index=False).mean()
print df[['Pregnancies', 'Outcome']].groupby(['Outcome'], as_index=False).mean()
print df[['DiabetesPedigreeFunction', 'Outcome']].groupby(['Outcome'], as_index=False).mean()
print df[['Insulin', 'Outcome']].groupby(['Outcome'], as_index=False).mean()

drop_elements = ['BloodPressure', 'SkinThickness', 'Insulin', 'Outcome']
# X_train = X_train.drop(drop_elements, axis=1)

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.linear_model import LogisticRegression

classifiers = [
    KNeighborsClassifier(3),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LDA(),
    QDA(),
    LogisticRegression()
]


log_cols = ["Classifier", "Accuracy"]
log = pd.DataFrame(columns=log_cols)

X = X_train
y = y_train

sss = StratifiedShuffleSplit(y, n_iter=10, test_size=0.1, random_state=0)

acc_dict = {}

for train_index, test_index in sss:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(X_train, y_train)
        train_predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, train_predictions)
        if name in acc_dict:
            acc_dict[name] += acc
        else:
            acc_dict[name] = acc
        # print '{0}: {1}'.format(name, acc * 100)

for clf in acc_dict:
    acc_dict[clf] = acc_dict[clf] / 10.0
    log_entry = pd.DataFrame([[clf, acc_dict[clf]]], columns=log_cols)
    log = log.append(log_entry)

plt.xlabel('Accuracy')
plt.title('Classifier Accuracy')
# plt.show()
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
from operator import itemgetter

sorted_dict = sorted(acc_dict.items(), key=itemgetter(1), reverse=True)

for k, v in sorted_dict:
    print "{0}-{1:.2%}".format(k, v)