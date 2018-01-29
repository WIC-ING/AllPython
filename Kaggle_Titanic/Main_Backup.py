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








#导入训练数据集并显示相关数据
data_train = pd.read_csv('train.csv')
print(data_train.info())
# print(data_train)


#用其他已知数据预测年龄的缺失值，并将数据补上
data_train, rfr = set_missing_ages(data_train)
#将数据集中船舱号这个特征转换为：有/无
data_train = set_Cabin_type(data_train)

# print(data_train)
# print(data_train.info())

#将标称型数据转换为独热编码
dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)

df = df.drop(['Pclass', 'Cabin', 'Embarked', 'Sex', 'Ticket', 'Name'], axis=1)
# print(df.info())


import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)

fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1), fare_scale_param)

# print(df)
#--------------------------------------
#---------------df为处理后的数据---------
#---------------通过df构建分类器---------
#--------------------------------------
from sklearn import linear_model
from sklearn.svm import SVC
# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
# clf = SVC(kernel='rbf', random_state=1, gamma=1.0, C=1.0,)
clf.fit(X, y)

# print(clf)

# 对训练数据集采用同样的处理方式
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
# print(data_test.info())


#将船舱号这一特征转换为是否为缺失值： yes or no
data_test = set_Cabin_type(data_test)
# print(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1,1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1,1), fare_scale_param)

# 用正则取出我们要的属性值
test_df = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# test_df = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Sex_.*|Pclass_.*')
test_np = test_df.as_matrix()


predictions = clf.predict(test_np)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("logistic_regression_predictions.csv", index=False)

print(pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)}, columns=['columns', 'coef']))





# #------------------------------------
# #----------BaseLine之后的进阶提升------
# #------------------------------------
# from sklearn import model_selection
#
# #  #简单看看打分情况
# # clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# # all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# # X = all_data.as_matrix()[:,1:]
# # y = all_data.as_matrix()[:,0]
# # print(model_selection.cross_val_score(clf, X, y, cv=5))
#
#
# # 分割数据，按照 训练数据:cv数据 = 7:3的比例
# split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
#
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
# clf.fit(train_df.as_matrix()[:,1:], train_df.as_matrix()[:,0])
#
# # 对cross validation数据进行预测
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:,1:])
#
# origin_data_train = pd.read_csv("train.csv")
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
#
# print(bad_cases)
# print('number of bad cases', bad_cases.shape[0])
# print('number of test cases', split_cv.shape[0])