import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

kf = KFold(n_splits=2)
kf.get_n_splits(X)

print(kf)

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]



#
# data = {'a':[1,2,3], 'b':[4,5,6]}
#
# data1 = {'key1':['a', 'a', 'b', 'b', 'a'],
#          'key2':['one', 'two', 'one', 'two', 'one'],
#          'data1':[1,2,3,4,5],
#          'data2':[6,7,8,9,10]}
#
#
# df = DataFrame(data1, columns=['data1', 'data2', 'key1', 'key2'])
#
#
#
# print(df, '\n')
#
# from sklearn.preprocessing import LabelEncoder
#
# class_le = LabelEncoder()
# y = class_le.fit_transform(df['key1'].values)
# df['key1'] = class_le.fit_transform(df['key1'].values)
# print(df)
#
# print(pd.get_dummies(df))




# print(df.dtypes)
# grouped = df.groupby(df.dtypes, axis=1)
# print(dict(list(grouped)))
