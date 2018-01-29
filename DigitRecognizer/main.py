import pandas as pd

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

y = train.iloc[:,0]
X = train.iloc[:,1:]

print(len(test))
print('Load Data has completed')



from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# X = np.array([[-1,-1],
#               [-2,-1],
#               [-3,-2],
#               [ 1, 1],
#               [ 2, 1],
#               [ 3, 2]
#               ])
#
# y = np.array([1,1,1,0,0,0])

print('Start Fitting\n\n')
nbrs = KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', n_jobs=-1).fit(X, y)

print('Start Predicting\n')
Label = nbrs.predict(test)

ResultSubmission = pd.DataFrame({
    'ImageId':list(range(1,28001)),
    'Label'  :Label
})

ResultSubmission.to_csv('Result.csv', index=False)

print('Over')
