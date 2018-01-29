import pandas as pd
import numpy as np

# df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
# print(df)
# df.to_csv('wdbc.txt')

df = pd.read_csv('wdbc.data', header=None)
# print(df.values)
#
# print('\n\n')

#-----------------------------------------------
#对数据集进行提取，并对labels进行编码
#-----------------------------------------------
from sklearn.preprocessing import LabelEncoder
X = df.iloc[1:,3:].values
y = df.iloc[1:, 2].values
le = LabelEncoder()
y = le.fit_transform(y)
# print(np.unique(y))



#-----------------------------------------------
#对数据集进行进行划分，提出训练集和测试集
#-----------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



#--------------------------------------------------------------
#网格法暴力搜索最优参数
#--------------------------------------------------------------
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
pipe_svc = Pipeline([ ('scl', StandardScaler()),
                      ('clf', SVC(random_state=1)) ])

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{
    'clf__C' : param_range,
    'clf__kernel' : ['linear']},
              {
    'clf__C' : param_range,
    'clf__gamma'  : param_range,
    'clf__kernel' : ['rbf']}]


from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=[
        {'max_depth' : [1,2,3,4,5,6,7,None]}],
    scoring='accuracy',
    cv=5)


scores = cross_val_score(gs,
                         X_train,
                         y_train,
                         scoring='accuracy',
                         cv=5)
print('CV accuracy: %.3f +/- %.3f' %
      (float(np.mean(scores)), float(np.std(scores))))

# gs = GridSearchCV(estimator=pipe_svc,
#                   param_grid=param_grid,
#                   scoring='accuracy',
#                   cv=10,
#                   n_jobs=-1)
#
#
# scores = cross_val_score(gs, X, y, scoring='accuracy', cv=5)
#
# print(scores.shape)
#
# print('CV accuracy: %.3f +/- %.3f' %
#       (float(np.mean(scores)), float(np.std(scores))))


# gs = gs.fit(X_train, y_train)
#
# print(gs.best_score_)
#
# print(gs.best_params_)
#
# clf = gs.best_estimator_
#
# clf.fit(X_train, y_train)
#
# print('Test accuracy: %.3f' % clf.score(X_test, y_test))

# # --------------------------------------------------------------
# # 用管道将	StandardScaler,	PCA和LogisticRegression连接起来，一起操作
# # --------------------------------------------------------------
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
#
# pipe_lr = Pipeline([('scl', StandardScaler()),
#                     ('pca', PCA(n_components=2)),
#                     ('clf', LogisticRegression(random_state=1))])
#
# pipe_lr.fit(X_train, y_train)
#
# print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))



#--------------------------------------------------------------
#------------------------画出方差，偏差曲线图---------------------
#--------------------------------------------------------------
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
#
# pipe_lr = Pipeline([
#                     ('scl', StandardScaler()),
#                     ('clf', LogisticRegression(
#                         penalty='l2', random_state=0))])
#
# train_sizes, train_scores, test_scores = learning_curve(estimator=pipe_lr,
#                                                         X=X_train,
#                                                         y=y_train,
#                                                         train_sizes=np.linspace(0.1,1,10),
#                                                         cv=10,
#                                                         n_jobs=1)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std  = np.std(train_scores, axis=1)
# test_mean  = np.mean(test_scores, axis=1)
# test_std   = np.std(test_scores, axis=1)
#
#
# plt.plot(train_sizes, train_mean,
#          color='blue', marker='o',
#          markersize=5,
#          label='training accuracy')
#
# plt.fill_between(train_sizes,
#                  train_mean+train_std,
#                  train_mean-train_std,
#                  alpha=0.15, color='blue')
#
#
# plt.plot(train_sizes, test_mean,
#          color='green', marker='o',
#          markersize=5,
#          label='training accuracy')
#
# plt.fill_between(train_sizes,
#                  test_mean+test_std,
#                  test_mean-test_std,
#                  alpha=0.15, color='green')
#
#
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.show()


#--------------------------------------------------------------
#------------------------K折交叉验证-----------------------------
#--------------------------------------------------------------
# from sklearn.model_selection import StratifiedKFold
#
#
#
# skf = StratifiedKFold(n_splits=10)
# skf.get_n_splits(X, y)
#
# scores = []
# k=0
# for train, test in skf.split(X, y):
#     k+=1
#     pipe_lr.fit(X[train], y[train])
#     score = pipe_lr.score(X[test], y[test])
#     scores.append(score)
#     print('Fold: %s, Class dist.: %s, Acc: %.3f' % (k+1, np.bincount(y[train]), score))
#
# print('CV accuracy: %.3f +/- %.3f' % (float(np.mean(scores)), float(np.std(scores))))