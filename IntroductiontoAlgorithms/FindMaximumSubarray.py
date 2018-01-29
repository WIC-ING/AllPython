a = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]

b = [-1, 1, 2, 3, 4]

c = [1,2]


def FindMaxCrossingSubArray(a, low, mid, high):
    maxLeft = maxRight = mid

    leftSum = float('-inf')
    Sum = 0
    for i in range(mid, low-1, -1):
        Sum = Sum + a[i]
        if Sum > leftSum:
            leftSum = Sum
            maxLeft = i


    rightSum = float('-inf')
    Sum = 0
    for i in range(mid+1, high+1):
        Sum = Sum + a[i]
        if Sum > rightSum:
            rightSum = Sum
            maxRight = i

    if leftSum == float('-inf') and rightSum != float('-inf'):
        return (maxLeft, maxRight, rightSum)
    elif leftSum != float('-inf') and rightSum == float('-inf'):
        return (maxLeft, maxRight, leftSum)
    else:
        return (maxLeft, maxRight, leftSum + rightSum)

def FindMaximumSubarray(a, low, high):
    if high == low:
        return (low, high, a[low])

    else:
        mid = int((low + high)/2)

        (leftLow, leftHigh, leftSum) = FindMaximumSubarray(a, low, mid)


        (rightLow, rightHigh, rightSum) = FindMaximumSubarray(a, mid+1, high)


        (crossLow, crossHigh, crossSum) = FindMaxCrossingSubArray(a, low, mid, high)

        if leftSum >= rightSum and leftSum >= crossSum:
            return (leftLow, leftHigh, leftSum)
        elif rightSum >= leftSum and rightSum >= crossSum:
            return (rightLow, rightHigh, rightSum)
        else:
            return (crossLow, crossHigh, crossSum)

# print(FindMaxCrossingSubArray(a,0,7,15))
# print(FindMaxCrossingSubArray(c,0,2,3))
print(FindMaximumSubarray(a,0,len(a)-1))

# print(FindMaximumSubarray(c,0,len(c)-1))
