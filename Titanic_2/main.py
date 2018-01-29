import pandas as pd
import numpy as np

input_df = pd.read_csv('train.csv', header=0)
submit_df = pd.read_csv('test.csv', header=0)

import warnings
warnings.filterwarnings('ignore')

#---------------------------------------
#------------对数据进行预处理--------------
#---------------------------------------


#合并数据
df = pd.concat([input_df, submit_df], ignore_index=True)
# df = input_df


#处理缺失值
df_Age = df.loc[df['Age'].notnull(), 'Age']
average_age   = df['Age'].mean()
std_age       = df['Age'].std()
count_nan_age = df['Age'].isnull().sum()
rand_age = np.random.randint(average_age-std_age, average_age+std_age, size=count_nan_age)

df.loc[df['Age'].isnull(), 'Age'] = rand_age

df.loc[df['Cabin'].isnull(), 'Cabin'] = 'U0'

df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].mean()

df.loc[df['Embarked'].isnull(), 'Embarked'] = df['Embarked'].dropna().mode().values

# print(df.info())


#将离散值转换为独热编码
embark_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
df = df.join(embark_dummies)
df.drop(['Embarked'], axis=1, inplace=True)


Pclass_dummies = pd.get_dummies(df['Pclass'], prefix='Pclass')
df = pd.concat([df, Pclass_dummies], axis=1)
df.drop('Pclass', axis=1, inplace=True)


#对Cabin值的处理
import re
df['CabinLetter'] = df['Cabin'].map( lambda x: re.compile('([a-zA-Z]+)').search(x).group() )
df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]

#对Sex值的处理
df.loc[df['Sex']=='female','Sex'] = 0
df.loc[df['Sex']=='male','Sex'] = 1


#对连续值进行标准化处理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1))

# #对票价特征进行分段处理
# df['Fare_bin'] = pd.qcut(df['Fare'], 4)
# df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]
# # df = pd.concat( [df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))] , axis=1 )
# Farebin_dummies = pd.get_dummies(df['Fare_bin_id'], prefix='Farebin')
# df = pd.concat( [df, Farebin_dummies], axis=1 )


# print(df)
#---------------------------------------
#---------------提取特征-----------------
#---------------------------------------

#-----------提取特征 --- 称呼（Title）-----------------
#处理称谓
p=re.compile(', (.*?)\.')
df['Title'] = df['Name'].map(lambda x: p.findall(x)[0])

# Group low-occuring, related titles together
df.loc[df['Title'] == 'Jonkheer', 'Title'] = 'Master'
df.loc[df['Title'].isin(['Ms','Mlle']), 'Title'] = 'Miss'
df.loc[df['Title'] == 'Mme', 'Title'] = 'Mrs'
df.loc[df['Title'].isin(['Capt', 'Don', 'Major', 'Col', 'Sir']), 'Title'] = 'Sir'
df.loc[df['Title'].isin(['Dona', 'Lady', 'the Countess']), 'Title'] = 'Lady'
df['Title_id'] = pd.factorize(df['Title'])[0]+1
df = df.drop(['Title'], axis=1)

TitleId_dummies = pd.get_dummies(df['Title_id'], prefix='TitlId')
df = pd.concat([df, TitleId_dummies], axis=1)

# df.drop(['Name'],axis=1,inplace=True)

# Title_dummies = pd.get_dummies(df['Title'], prefix='Title')
# df = pd.concat( [df, Title_dummies], axis=1 )


#-----------提取特征 --- 甲板（Deck）-----------------
# p = re.compile('([a-zA-Z]+)')
df = df.rename(index=str, columns={'CabinLetter':'Deck'})
Deck_dummies = pd.get_dummies(df['Deck'], prefix='Deck')
df = pd.concat([df, Deck_dummies], axis=1)


#-----------提取特征 --- 房间（Room）-----------------
p = re.compile('([0-9]+)')
df['Room'] = df['Cabin'].map( lambda x : int(p.search(x).group()) + 1 )
df.drop(['Cabin'], axis=1, inplace=True)

#-----------提取特征 --- 孩子（Child）-----------------
df.loc[df['Age']<=16, 'Child'] = 1
df.loc[df['Age']>16, 'Child'] = 0


#-----------提取特征 --- 是否有家人（WithFamily）--------
df['WithFamily'] = df['Parch'] + df['SibSp']
df.loc[df['WithFamily']>=1, 'WithFamily'] = 1
df.loc[df['WithFamily']==0, 'WithFamily'] = 0

#-----------提取特征 --- 姓氏（Surname）----------------
df['Surname'] = df['Name'].map(lambda x: re.compile("(Mr|Mrs|Miss|Master|Don|Rev|Dr|Mme|Ms|Major|Lady|Sir|Mlle|Col|Capt|the Countess|Jonkheer|Dona)\.\s(\w*)").findall(x)[0][1])
df['Surname'] = pd.factorize(df['Surname'])[0]

df.drop('Name', axis=1, inplace=True)

# print(df['Surname'])
#处理家庭大小
df['FamilySize'] =  df["Parch"] + df["SibSp"] + 1
df.loc[df['FamilySize'] < 3, 'FamilySize'] = 'small'
df.loc[df['FamilySize'] != 'small', 'FamilySize'] = 'big'
df.loc[df['FamilySize'] == 'small', 'FamilySize'] = 0
df.loc[df['FamilySize'] == 'big',   'FamilySize'] = 1
df['FamilySize'] = df['FamilySize'].astype(int)



#--------------------------------------------------
#-------------------训练分类器-----------------------
#--------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
from sklearn import model_selection
from operator import *

X_test_original = df[input_df.shape[0]:]

df = df.filter(regex='Survived|TitlId_.*|Sex|Fare_.*|Age_.*|Surname|Pclass_.*|Embarked_.*|WithFamily')

cols = list(df)
cols.insert(0, cols.pop(cols.index('Survived')))
df = df.ix[:, cols]

print(df.info())

X = df[:input_df.shape[0]].values[:,1::]
y = df[:input_df.shape[0]].values[:,0]
y = [int(data) for data in y]


X_test = df[input_df.shape[0]:].values[:,1::]


sqrtfeat = int(np.sqrt(X.shape[1]))
minsampsplit = int(X.shape[0]*0.015)

def report(grid_scores, n_top=5):
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print('Parameters with rank: {0}'.format(i+1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
            score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

        if params == None:
            params = score.parameters
    return params

grid_test1 = {"n_estimators"      :  [30000],
               "criterion"         : ["gini", "entropy"],
               "max_features"      : [sqrtfeat],
               "max_depth"         : [5, 10, 25],
               "min_samples_split" : [2, 5, 10, minsampsplit ] }

forest = RandomForestClassifier(oob_score=1)

print("Hyperparameter optimization using GridSearchCV...")
grid_search = model_selection.GridSearchCV(forest, grid_test1, n_jobs=-1, cv=10)

grid_search.fit(X, y)
Y_pred = grid_search.predict(X_test)

print(grid_search.score(X, y))


#
# random_forest = RandomForestClassifier(oob_score=True, n_estimators=30000,max_depth=25, n_jobs=-1)
# random_forest.fit(X,y)
#
#
# Y_pred = random_forest.predict(X_test)
# print(random_forest.score(X, y))

ResultSubmission = pd.DataFrame({
    'PassengerId':list(X_test_original['PassengerId']),
    'Survived'   :Y_pred
})

ResultSubmission.to_csv('Result.csv', index=False)


