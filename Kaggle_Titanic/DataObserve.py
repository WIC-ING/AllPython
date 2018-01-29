import pandas as pd
import numpy as np


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

    # df['CabinNum'] = [int(data) for data in list(df['CabinNum'].values/20)]

    return df






data_train = pd.read_csv('train.csv')
data_test  = pd.read_csv('test.csv')

s = data_train['Name'].iloc[0]

import re

p= re.compile(", (.*)\.")

data_train['title'] = data_train['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

print(np.unique(data_train['title'].values))
# print(data_train.loc[data_train['title']=='the', ])


# data_train = set_CabinNum(data_train)
# data_train = set_motherlabel(data_train)
#
# print(data_train['mother'].describe())
#
# data_hasCabin = data_train.loc[data_train['Cabin'].notnull()]
# data_notCabin = data_train.loc[data_train['Cabin'].isnull()]
#
# data_survived    = data_hasCabin.loc[data_hasCabin['Survived']==1]
# data_notsurvived = data_hasCabin.loc[data_hasCabin['Survived']==0]

# print(data_survived.info())
# print(data_survived['CabinNum'].describe())
#
# print('\n\n')
#
# print((data_notsurvived.info()))
# print(data_notsurvived['CabinNum'].describe())

# data_train['CabinNum'].to_csv('CabinNum_of_data_train', index=False)

# print(data_train['CabinNum'].describe())



# print('\n\n')
#
# print(data_test['FamilySize'].value_counts())
# print(data_train['mother'].value_counts())

# print(data_temp.info())
# #
# # print(data_train.describe())
#
#
#
#
# data_temp.loc[data_temp['Cabin'].notnull(), 'CabinLabel'] = [str[0] for str in data_temp[data_temp['Cabin'].notnull()]['Cabin']]
# data_temp.loc[data_temp['Cabin'].isnull(), 'CabinLabel'] = 'Z'
# data_temp.loc[data_temp['CabinLabel'] == 'T', 'CabinLabel'] = 'Z'
# print(set(data_temp['CabinLabel']))
# print(data_test['CabinLabel'].value_counts())



#------------------------------------
#---------对年龄分布的观察--------------
#------------------------------------
# train_hasAge = data_train[data_train['Age'].notnull()]
# print(train_hasAge['Age'].value_counts())
#
# train_hasAge['Age'] = [int(data) for data in (train_hasAge['Age'].values/10)]
#
# result = pd.DataFrame([])
# for i in sorted(train_hasAge['Age'].values):
#      result[str(i)] = train_hasAge.loc[train_hasAge['Age']==i, 'Survived'].value_counts()
# result[result.isnull()] = 0
# result = result.T
#
# #
# result['survived rate'] = result[1]/(result[1]+result[0])
# print(result)
#
# import matplotlib.pyplot as plt
#
# # train_hasAge['Age'].hist()
#
# result['survived rate'].plot(kind='bar', stacked=True)
# plt.show()




# import matplotlib.pyplot as plt
# fig = plt.figure(figsize=(20,10), dpi=100)
# # fig.set(alpha=0.2)
#
# plt.subplot2grid((2,3),(0,0))             # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"获救情况 (1为获救)") # 标题
# plt.ylabel(u"人数")
#
# plt.subplot2grid((2,3),(0,1))
# data_train.Pclass.value_counts().plot(kind="bar")
# plt.ylabel(u"人数")
# plt.title(u"乘客等级分布")
#
# plt.subplot2grid((2,3),(0,2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u"年龄")                         # 设定纵坐标名称
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"按年龄看获救分布 (1为获救)")
#
# plt.subplot2grid((2,3),(1,0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(kind='kde')
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u"年龄")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"各等级的乘客年龄分布")
# plt.legend((u'头等舱', u'2等舱',u'3等舱'),loc='best') # sets our legend for our graph.
#
#
# plt.subplot2grid((2,3),(1,2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u"各登船口岸上船人数")
# plt.ylabel(u"人数")
# plt.show()



# #看看各乘客等级的获救情况
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
#
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
#
# df.plot(kind='bar', stacked=True)
# plt.title(u"各乘客等级的获救情况")
# plt.xlabel(u"乘客等级")
# plt.ylabel(u"人数")
# plt.show()


# #看看各性别的获救情况
# Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
# Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()

# sex_Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
# sex_Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
#
# print(sex_Survived_0)
#
# df=pd.DataFrame({u'获救':sex_Survived_1, u'未获救':sex_Survived_0})
# # df=df.T
# print(df)
# df.plot(kind='bar', stacked=True)
# plt.title(u"按性别看获救情况")
# plt.xlabel(u"性别")
# plt.ylabel(u"人数")
# plt.show()






#然后我们再来看看各种舱级别情况下各性别的获救情况
# fig=plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
# plt.title(u"根据舱等级和性别的获救情况")
#
# ax1=fig.add_subplot(141)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female highclass", color='#FA2479')
# ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
# ax1.legend([u"女性/高级舱"], loc='best')
#
# ax2=fig.add_subplot(142, sharey=ax1)
# data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='female, low class', color='pink')
# ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"女性/低级舱"], loc='best')
#
# ax3=fig.add_subplot(143, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label='male, high class',color='lightblue')
# ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/高级舱"], loc='best')
#
# ax4=fig.add_subplot(144, sharey=ax1)
# data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
# plt.legend([u"男性/低级舱"], loc='best')
#
# plt.show()


# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数
#
# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# df=pd.DataFrame({u'获救':Survived_1, u'未获救':Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u"各登录港口乘客的获救情况")
# plt.xlabel(u"登录港口")
# plt.ylabel(u"人数")
#
# plt.show()


# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
# g = data_train.groupby(['SibSp','Survived'])
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)
#
# # print(data_train.Cabin.value_counts())
#
# fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数



# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df=pd.DataFrame({u'有':Survived_cabin, u'无':Survived_nocabin}).transpose()
# df.plot(kind='bar', stacked=True)
# plt.title(u"按Cabin有无看获救情况")
# plt.xlabel(u"Cabin有无")
# plt.ylabel(u"人数")
# plt.show()
