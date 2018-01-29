




# import pandas as pd
# from pandas import DataFrame, Series
#
# data = {'year':[200,300,400], 'income':[12,13,14]}
#
# df = DataFrame(data)
#
# print(df, '\n\n')
#
# temp_data = df.to_dict(orient='list')
#
# # print(temp_data)
#
# df1 = DataFrame(temp_data, columns=['year', 'income'], index=[2,1,0])
#
# print(df1)
#
# df1.loc[0] = [5,6]
# print(df1)

# df1['year'] = [1,2,3]
#
# print(df1)
















# import pandas as pd
#
# temp_result = pd.read_csv('')
#
# pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

# data_train = pd.read_csv('train.csv')
# df = data_train
# # print(type(data_train.Age.isnull()))
#
# age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
#
# data = age_df.iloc[0:10]
#
# data = data.drop(['Age', 'Fare'], axis=1)
#
# print(data)



# import pandas as pd
#
# data_test = pd.read_csv('test.csv')
#
# print(data_test.info())

# data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
#
#
# #接着我们对test_data做和train_data中一致的特征变换
# #接着用同样的RandomForestRegressor模型填上丢失的年龄
# temp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
#
# null_age = temp_df[data_test.Age.isnull()].as_matrix()
#
#
# #根据特征X预测年龄并补上
# x = null_age[:, 1:]
# predictedAge = rfr





# from sklearn.preprocessing import StandardScaler
#
# data = [[0], [0], [1], [1]]
# scaler = StandardScaler()
# # print(scaler.fit(data))
#
# # print(scaler.mean_)
#
# print(scaler.fit_transform(data))




# print(scaler.transform([[2, 2]]))

# import pandas as pd
# from pandas import DataFrame, Series
#
# data = {'x1':[15,10], 'x2':[15,20], 'x3':[1,2]}
#
# df = DataFrame(data)
#
# print(df[['x1', 'x3']].as_matrix())


# import matplotlib.pyplot as plt
#
# plt.figure()
# df.plot(kind='bar', stacked=True)
# plt.show()
# print(df)
