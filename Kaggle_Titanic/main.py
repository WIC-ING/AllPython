import pandas as pd
import numpy as np

# print(data_train.describe())

# print(data_train)


from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):

    # 将已有的数据型特征取出来丢进Random Forest Regression中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age   = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:,0]

    # x即特征属性值
    X = known_age[:,1:]

    # fit到RandomForestRegression之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X,y)

    #用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = 'Yes'
    df.loc[ (df.Cabin.isnull()),  'Cabin' ] = 'No'
    return df

def set_child_type(df):
    df.loc[data_train['Age'] <= 12, 'child'] = 'yse'
    df.loc[data_train['Age'] > 12, 'child'] = 'no'
    return df

def set_motherlabel(df):
    df.loc[(df['Sex'] == 'female') & (df['Parch']>0), 'mother'] = 'yes'
    df.loc[(df['Sex'] == 'male') | (df['Parch']==0), 'mother'] = 'no'
    return df

def set_FamilySize(df):
    df['FamilySize'] = df['Parch'] + df['SibSp']
    return df


def set_CabinNum(df):

    import re
    p = re.compile(r'\d+')

    df_hasCabin = df.loc[df['Cabin'].notnull()]

    CabinNum = []
    for str in list(df_hasCabin['Cabin']):
        tempList = [data for data in str.split()]
        n = len(tempList)

        tempNum = []
        for str in tempList:
            tempNum.extend(p.findall(str))
        tempNum = [int(s) for s in tempNum]
        CabinNum.append(int(sum(tempNum) / n))

    df.loc[df['Cabin'].notnull(), 'CabinNum']  = CabinNum
    df.loc[df['Cabin'].isnull(),  'CabinNum']  = 0

    df['CabinNum'] = [int(data) for data in list(df['CabinNum'].values/75)]

    return df




#导入训练数据集并显示相关数据
data_train = pd.read_csv('train.csv')
print(data_train.info())
# print(data_train)


#用其他已知数据预测年龄的缺失值，并将数据补上
data_train, rfr = set_missing_ages(data_train)


#建立特征 -- CabinNum
data_train = set_CabinNum(data_train)

#设定是否为母亲标签
# data_train = set_motherlabel(data_train)

#将数据集中船舱号这个特征转换为：有/无
data_train = set_Cabin_type(data_train)

#建立特征 -- FamilySize
data_train = set_FamilySize(data_train)



#将标称型数据转换为独热编码
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df = df.drop(['Pclass', 'Cabin', 'Embarked', 'Sex', 'Ticket', 'Name'], axis=1)



import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

# print(df['CabinNum'].value_counts())
#--------------------------------------
#---------------df为处理后的数据---------
#---------------通过df构建分类器---------
#--------------------------------------
from sklearn import linear_model
from sklearn.svm import SVC
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

print(train_df.info())

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
#clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf = SVC(kernel='poly', degree=2,random_state=1, gamma=1.0, C=20.0,)
clf.fit(X, y)

# print(clf)




# #-------------------------------------------
# #--------------画Learning Curve-------------
# #-------------------------------------------
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
#
# # 用Sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
# def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
#                         train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
#     """
#
#     画出data在某模型上的learning curve
#     --------------------
#     :param esrimator: 所用分类器
#     :param title:     图表的标题
#     :param X:         输入的feature, numpy类型
#     :param y:         输入的target vector
#     :param ylim:      tuple格式的(ylim, ymax), 设定图像中纵坐标的最低点和最高点
#     :param cv:        做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training（默认为3份）
#     :param n_jobs:    并行的任务数
#     :param train_sizes:
#     :param verbose:
#     :param plot:
#     :return:
#     """
#
#     train_sizes, train_scores, test_scores = learning_curve(
#         estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose
#     )
#
#     train_scores_mean = np.mean(train_scores, axis=1)
#     train_scores_std  = np.std(train_scores, axis=1)
#
#     test_scores_mean  = np.mean(test_scores, axis=1)
#     test_scores_std   = np.std(test_scores, axis=1)
#
#     if plot:
#         plt.figure()
#         plt.title(title)
#         if ylim is not None:
#             plt.ylim(*ylim)
#         plt.xlabel(u'训练样本数')
#         plt.ylabel(u'得分')
#
#         plt.gca().invert_yaxis()
#         plt.grid()
#
#         plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
#                          alpha=0.1, color="b")
#         plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
#                          alpha=0.1, color="r")
#         plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
#         plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")
#
#         plt.legend(loc="best")
#
#         plt.draw()
#         plt.show()
#         plt.gca().invert_yaxis()
#     midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
#     diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
#     return midpoint, diff
#
# plot_learning_curve(clf, u"学习曲线", X, y)

#------------------------------------
#----------BaseLine之后的进阶提升------
#------------------------------------
# from sklearn import model_selection
#
# #  #简单看看打分情况
# # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# # all_data = df.filter(regex='Survived|Age_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize')
# # X = all_data.as_matrix()[:,1:]
# # y = all_data.as_matrix()[:,0]
# # print(model_selection.cross_val_score(clf, X, y, cv=5))
#
#
# # # 分割数据，按照 训练数据:cv数据 = 7:3的比例
# split_train, split_cv = model_selection.train_test_split(df, test_size=0.2, random_state=1)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize|CabinNum')
#
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
#
# # 对cross validation数据进行预测
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|FamilySize|CabinNum')
# predictions = clf.predict(cv_df.as_matrix()[:,1:])
#
# origin_data_train = pd.read_csv("train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
#
# # print(bad_cases)
# bad_cases.to_csv("bad_cases.csv", index=False)
# print('\n\n', pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}, columns=['columns', 'coef']))
#
# print('\n\nnumber of bad cases', bad_cases.shape[0])
# print('number of test cases', split_cv.shape[0])
# print('accuracy of test is :', (split_cv.shape[0] - bad_cases.shape[0])/ split_cv.shape[0])

# #-----------------------------------------------------
# #---------------# 对训练数据集采用同样的处理方式-----------
# #---------------通过分类器对数据进行预测------------------
# #----------------------------------------------------

data_test = pd.read_csv('test.csv')
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0



# #接着我们对test_data做和train_data中一致的特征变换
#接着用同样的RandomForestRegressor模型填上丢失的年龄
temp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = temp_df[data_test.Age.isnull()].as_matrix()
# #根据特征X预测年龄并补上
X = null_age[:, 1:]
predictedAge = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAge

# print(data_test)

#建立特征 -- CabinNum
data_test = set_CabinNum(data_test)

#将船舱号这一特征转换为是否为缺失值： yes or no
data_test = set_Cabin_type(data_test)

#建立特征 -- FamilySize
data_test = set_FamilySize(data_test)

#进行独热编码
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

# # 用正则取出我们要的属性值
# test_df = df_test.filter(regex='Age_.*|SibSp|Cabin_.*|Embarked_S|Sex_.*|Pclass_1|Pclass_3|FamilySize|CabinNum')
# # test_df = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# test_np = test_df.as_matrix()
# # print('\n\n info of test_df:')
# # print(test_df.info())
#
# predictions = clf.predict(test_np)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
# result.to_csv("logistic_regression_predictions.csv", index=False)
#
# print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}, columns=['columns', 'coef']))




#-----------------------------------------------
#----------------模型融合------------------------
#-----------------------------------------------

# from sklearn.ensemble import BaggingRegressor
#
# train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
# train_np = train_df.as_matrix()
#
# # y即Survival结果
# y = train_np[:, 0]
#
# # X即特征属性值
# X = train_np[:, 1:]
#
# # fit到BaggingRegressor之中
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
# bagging_clf.fit(X, y)
#
# test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = bagging_clf.predict(test)
# result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
# result.to_csv("logistic_regression_bagging_predictions.csv", index=False)