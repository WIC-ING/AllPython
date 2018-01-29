import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#导入训练集数据
df_train = pd.read_csv('train.csv')
# print(df_train.info())


#处理缺失值 - MISSING DATA -- STEP5
total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
# print(missing_data.head(19).index)
# print(missing_data.head(19))
df_train = df_train.drop(missing_data.loc[missing_data['Total']>1].index, axis=1)
df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index, axis=0)
# print(df_train.info())


#输出特征为（地上最大面积）中值最大的两条数据的Id并删除他们
# print(df_train.sort_values(by = 'GrLivArea', ascending = False)[:2][['Id', 'GrLivArea']])
df_train = df_train.drop(df_train.loc[df_train['Id']==1299].index)
df_train = df_train.drop(df_train.loc[df_train['Id']==524].index)

#标准化数据 -- SETP6
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:, np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
# print('outer range (low) of the distribution:')
# print(low_range)
# print('\nouter range (high) of the distribution')
# print(high_range)

df_train['SalePrice'] = np.log(df_train['SalePrice'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0
df_train.loc[df_train['TotalBsmtSF']>0, 'HasBsmt'] = 1
df_train.loc[df_train['HasBsmt']==1, 'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])

df_train = pd.get_dummies(df_train)
print(df_train.info())



# sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
# plt.show()



# sns.distplot(df_train['TotalBsmtSF'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)
# plt.show()




#通过散点图矩阵，观察与目标特征相关性最强的特征之间的关系 -- STEP4
# corrmat = df_train.corr()
# cols = corrmat.nlargest(10, 'SalePrice')
# print(cols.index)
#
# sns.set()
# cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
# sns.pairplot(df_train[cols], size=2.5)
# plt.show()

# #计算相关矩阵,通过heatmap做出热力图， 观察不同特征之间的关系-- STEP3
# corrmat = df_train.corr()
# f, ax = plt.subplots(figsize=(12,9))
# sns.heatmap(corrmat, vmax=.8, square=True)
# # plt.show()
# #
# k = 10
# cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
# cm = np.corrcoef(df_train[cols].values.T)
# sns.set(font_scale=1.25)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':10}, yticklabels=cols.values, xticklabels=cols.values)
# plt.show()

# 观察数据一般特征与目标特征的相关性 -- STEP2
# var = '2ndFlrSF'
# data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
# # f, ax = plt.subplots(figsize=(8,6))
# # fig = sns.boxplot(x=var, y='SalePrice', data=data)
# # fig.axis(ymin=0, ymax=800000)
#
# data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
#
# plt.show()




#做出直方图以及核函数密度估计 -- STEP1
# sns.distplot(df_train['SalePrice'])
# # plt.show()


#计算房价数据的峰度以及偏度
# print('skewness: %f' % df_train['SalePrice'].skew())
# print('kurtosis: %f' % df_train['SalePrice'].kurt())



# print(df_train['SalePrice'].describe())

# print(df_train.columns)
