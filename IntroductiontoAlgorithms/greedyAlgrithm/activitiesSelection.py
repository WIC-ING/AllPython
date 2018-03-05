# 贪心算法之活动选择问题

# S表示不同活动的开始时间
S = [0, 1, 3, 0, 5, 3, 5,  6,  8,  8,  2, 12]
# F表示不同活动的结束时间
F = [0, 4, 5, 6, 7, 9, 9, 10, 11, 12, 14, 16]


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





print('一共有{}个活动'.format(len(S)-1))

print('最大兼容活动集为:{}'.format(sorted(list(  RECURSIVE_ACTIVITY_SELECTOR(S, F, 0, 11)  ))))
