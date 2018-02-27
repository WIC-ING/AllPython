# -*- coding: utf-8 -*-
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]


# 函数名称：cut_rod
# 函数功能：自顶向下的实现最优值的搜索
# 函数分析：时间复杂度较高，因为会不断重复计算底层结果
def cut_rod(prices, n):
    if n==0: return 0
    q = float('-inf')
    for i in range(1, n+1):
        q = max(q, prices[i] + cut_rod(prices, n-i))
    return q



# 函数名称：MEMOIZED_CUT_ROD
# 函数功能：带记忆版 -- 自顶向下的实现最优值的搜索
# 函数分析：时间复杂度为: n方
def MEMOIZED_CUT_ROD(prices, n):
    r = [float('-inf')] * (n+1)
    return MEMOIZED_CUT_ROD_AUX(prices, n, r)

def MEMOIZED_CUT_ROD_AUX(prices, n, r):
    if r[n]>=0:
        return r[n]
    q = float('-inf')
    if n==0:
        q = 0
    else:
        for i in range(1, n+1):
            q = max(q, prices[i] + MEMOIZED_CUT_ROD_AUX(prices, n-i, r))
    r[n] = q
    return q



def BOTTOM_UP_CUT_ROD(prices, n):
    r = [float('-inf')] * (n+1)
    r[0] = 0
    for j in range(1, n+1):
        q = float('-inf')
        for i in range(1, j+1):
            q = max(q, prices[i] + r[j-i])
        r[j] = q
    return r[n]

n = 5

print cut_rod(prices, n)
print MEMOIZED_CUT_ROD(prices, n)
print BOTTOM_UP_CUT_ROD(prices, n)