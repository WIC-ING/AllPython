import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('data.csv')

train = data[data['shot_made_flag'].notnull()]
test  = data[data['shot_made_flag'].isnull()].drop('shot_made_flag', axis=1)

print(train.info(), '\n')

print(train.describe(), '\n\n\n')

y = train['shot_made_flag']

#计算命中率
# HitRate = y.value_counts()[1.0]/len(y)
# print(HitRate)


#统计投篮类型与命中率之间的关系
# grouped = train['shot_made_flag'].groupby(train['action_type'])
# grouped.mean().plot(kind='bar')
# plt.show()

#统计混合投篮类型与命中率之间的关系
# grouped = train['shot_made_flag'].groupby(train['combined_shot_type'])
# grouped.mean().plot(kind='bar')
# plt.show()

#统计game_event_id与命中率之间的关系
# grouped = train['shot_made_flag'].groupby(train['game_id'])
# grouped.mean().plot(kind='bar')
# plt.show()


#观察lat与命中率之间的关系
plt.figure()
plt.scatter(train['lat'], train['shot_made_flag'])
plt.show()

print(len(set(train['game_id'].values)))

# print(test.info(), '\n')
#
# print(test.describe())



