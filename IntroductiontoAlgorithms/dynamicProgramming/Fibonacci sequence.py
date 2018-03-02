# -*- coding: utf-8 -*-

import time

# 函数名称：fibonacciSequence
# 函数功能：自底向上 -- 产生斐波切纳数列中的某一位
# 输入参数：int
# 返回参数：int
def fibonacciSequence(n):
    if n<=2: return 1

    seq = [1, 1]
    for i in range(2, n):
        seq.append(seq[-1] + seq[-2])
    return seq[-1]


# 函数名称：fibonacciSequence_up_to_buttom
# 函数功能：自顶向下 -- 产生斐波切纳数列中的某一位
# 输入参数：int
# 返回参数：int
def fibonacciSequence_up_to_buttom(n):
    if n<=2: return 1
    return fibonacciSequence_up_to_buttom(n-1) + fibonacciSequence_up_to_buttom(n-2)


n = 35

startTime = time.time()
print (fibonacciSequence(n))
endTime   = time.time()
print ('共耗时：{}s\n\n'.format(endTime-startTime))


startTime = time.time()
print (fibonacciSequence_up_to_buttom(n))
endTime   = time.time()
print ('共耗时：{}s'.format(endTime-startTime))

