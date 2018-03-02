# -*- coding:utf-8 -*-

# 函数名称：MATRIX_CHAIN_ORDER
# 函数功能：寻找矩阵链乘括号分割的最佳方案
# 输入参数：list[int]
# 返回参数：list[[int]], list[[int]]
def MATRIX_CHAIN_ORDER(p):
    n = len(p) - 1

    # m[i][j] 表示矩阵链中从第i个矩阵到第j个矩阵连乘所需要带的最小计算代价
    m = [[ 0 for i in range(n)] for j in range(n) ]
    s = [[-1 for i in range(n)] for j in range(n) ]

    for l in range(2, n+1):      # l为子问题中，矩阵链的长度
        for i in range(1, n-l+2):
            j = i+l-1
            m[i-1][j-1] = float('inf')
            for k in range(i, j):
                q = m[i-1][k-1] + m[k][j-1] + p[i-1]*p[k]*p[j]
                if q < m[i-1][j-1]:
                    m[i-1][j-1] = q
                    s[i-1][j-1] = k

    return m, s


# 函数名称：PRINT_OPTIMAL_PARENS
# 函数功能：输出矩阵链乘括号分割的最佳方案
# 输入参数：list[list[int]], int, int
# 返回参数：None
def PRINT_OPTIMAL_PARENS(s, i, j):
    # print 'begin', i, j,
    if i==j:
        print 'A{}'.format(i),
    else:

        print '(',
        PRINT_OPTIMAL_PARENS(s, i, s[i-1][j-1])
        PRINT_OPTIMAL_PARENS(s, s[i-1][j-1]+1, j)
        print ')',


p = [5, 10, 3, 12, 5, 50, 6]
p = [30, 35, 15, 5, 10, 20, 25]
m,s = MATRIX_CHAIN_ORDER(p)
print(s)
PRINT_OPTIMAL_PARENS(s, 1, 6)
