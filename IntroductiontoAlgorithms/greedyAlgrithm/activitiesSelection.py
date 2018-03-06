# -*- coding:utf-8 -*-
# 贪心算法之活动选择问题

import time

# S表示不同活动的开始时间
S = [0, 1, 3, 0, 5, 3, 5,  6,  8,  8,  2, 12,0]
# F表示不同活动的结束时间
F = [0, 4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16,0]


# 函数名称：RECURSIVE_ACTIVITY_SELECTOR
# 函数功能：递归贪心算法 之活动选择问题
# 输入参数：list[int], list[int], int, int
# 返回参数：set()
def RECURSIVE_ACTIVITY_SELECTOR(S, F, k, n):
    m = k+1
    while m<=n and S[m]<F[k]:
        m+=1
    if m<=n:
        setOfAct = set([m])
        for ele in RECURSIVE_ACTIVITY_SELECTOR(S, F, m, n):
            setOfAct.add(ele)
        return setOfAct
    else:
        return set()

# 函数名称：GREEDY_ACTIVITY_SELECTOR
# 函数功能：迭代贪心算法 之活动选择问题
# 输入参数：list[int], list[int]
# 返回参数：set()
def GREEDY_ACTIVITY_SELECTOR(S, F):
    n = len(S)
    A = set([1])
    k = 0
    for m in range(1, n):
        if S[m]>=F[k]:
            A.add(m)
            k=m
    return A


# 函数名称：dynamicProgramActSel
# 函数功能：动态规划算法 之活动选择问题
# 输入参数：list[int], list[int]
# 返回参数：list[list[]]
def dynamicProgramActSel(S, F):
    n = len(S)
    # setOfAct = set()
    r = [[0 for j in range(n)] for i in range(n)]

    for i in range(n-1):
        r[i][i+1] = 1


    for l in range(2, n):
        for i in range(n-l):
            j=i+l
            if F[i]<=S[j]:
                for k in range(i+1, j):
                    if S[k]<F[i] or F[k]>S[j]:
                        continue
                    temp = r[i][k] + r[k][j] + 1
                    if temp > r[i][j]:
                        r[i][j] = temp
    return r


print('一共有{}个活动'.format(len(S)-2))

start = time.time()
print('最大兼容活动集为:{}'.format(sorted(list(  RECURSIVE_ACTIVITY_SELECTOR(S, F, 0, 11)  ))))
print('用时：{}s'.format(time.time()-start))

start = time.time()
print('最大兼容活动集为:{}'.format(sorted(list(  GREEDY_ACTIVITY_SELECTOR(S, F)  ))))
print('用时：{}s'.format(time.time()-start))


print ('动态规划算法：')
# dynamicProgramActSel(S, F)
start = time.time()
for line in  dynamicProgramActSel(S, F):
    print (line)
print('用时：{}s'.format(time.time()-start))
