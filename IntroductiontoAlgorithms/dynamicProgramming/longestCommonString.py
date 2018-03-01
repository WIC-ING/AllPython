# -*- coding:utf-8 -*-

# 函数名称：LCS_LENGTH
# 函数功能：自底向上的计算各个子问题中，LCS的长度
# 输入参数：list[], list[]
# 返回翻出：list[list[]], list[list[]]
def LCS_LENGTH(X,Y):
    m = len(X)
    n = len(Y)
    print(m, n)
    b = [[None for i in range(n)] for j in range(m)]
    c = [[0 for i in range(n+1)] for j in range(m+1)]

    for i in range(1,m+1):
        for j in range(1,n+1):
            if X[i-1]==Y[j-1]:
                c[i][j] = c[i-1][j-1] + 1
                b[i-1][j-1] = 'lu'
            elif c[i-1][j]>=c[i][j-1]:
                c[i][j] = c[i-1][j]
                b[i-1][j-1] = 'up'
            else:
                c[i][j] = c[i][j-1]
                b[i-1][j-1] = 'lf'
    return c, b



def PRINT_LCS(b, X, i, j):
    if i==-1 or j==-1:
        return

    if b[i][j]=='lu':
        PRINT_LCS(b, X, i-1, j-1)
        print X[i],
    elif b[i][j]=='up':
        PRINT_LCS(b, X, i-1, j)
    else:
        PRINT_LCS(b, X, i, j-1)





X = [1,0,0,1,0,1,0,1]
Y = [0,1,0,1,0,1,1,0]

X = ['A', 'B', 'C', 'B', 'D', 'A', 'B']
Y = ['B', 'D', 'C', 'A', 'B', 'A']



c,b = LCS_LENGTH(X, Y)

print 'c&b:'
for line in c:
    print line

print '\n\n'

for line in b:
    print line

PRINT_LCS(b, X, 6, 5)
