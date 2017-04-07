## This code gives highest accuracy as given below

The updated code uses Xgboost to get **highest accuracy of 79.63%**

```
Accuracy: 79.63%
Thresh=0.339, n=1, Accuracy: 75.93%
Thresh=0.193, n=2, Accuracy: 74.07%
Thresh=0.184, n=3, Accuracy: 74.07%
Thresh=0.157, n=4, Accuracy: 75.93%
Thresh=0.128, n=5, Accuracy: 74.07%
```

~~This code uses feature selection and then uses all classifiers to get highest accuracy using LR as below~~

```
LogisticRegression-76.11%
LDA-75.74%
GaussianNB-74.81%
QDA-74.07%
RandomForestClassifier-72.78%
AdaBoostClassifier-72.41%
GradientBoostingClassifier-72.41%
KNeighborsClassifier-70.00%
DecisionTreeClassifier-63.70%
SVC-62.04%


```
