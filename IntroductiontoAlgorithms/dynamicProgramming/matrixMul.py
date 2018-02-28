# -*- coding:utf-8 -*-

# 函数名称：MATRIX_CHAIN_ORDER
# 函数功能：寻找矩阵链乘括号分割的最佳方案
# 输入参数：list[int]
# 返回参数：list[[int]], list[[int]]
def MATRIX_CHAIN_ORDER(p):
    n = len(p) - 1
    m = [[0]*n]*n
    s = [[0]*(n-1)]*(n-1)
    for l in range(2, n+1):      # l为子问题中，矩阵链的长度
        for i in range(1, n-l+2):
            j = i+l-1
            m[i][j] = float('inf')

            for k in range(i, j):
                q = m[i][k] + m[k+1][j] + p[i-1]*p[k]*p[j]
                if q < m[i][j]:
                    m[i][j] = q
                    s[i][j] = k

    return m, s


