from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()

X, y = iris.data[50:, [1, 2]], iris.target[50:]

le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np


clf1 = LogisticRegression(penalty='l2', C=0.001, random_state=0)

clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)

clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([('sc', StandardScaler()),
                  ('clf', clf1)])

pipe2 = Pipeline([('sc', StandardScaler()),
                  ('clf', clf2)])

pipe3 = Pipeline([('sc', StandardScaler()),
                  ('clf', clf3)])

clf_labels = ['LR', 'DecisionTree', 'KNN']

print('10-fold cross validation:\n')
for clf, label in zip([pipe1, pipe2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, scoring='roc_auc')
    print('ROC AUC: %0.2f (+/- %0.2f) [%s]'
          % (scores.mean(), scores.std(), label))

