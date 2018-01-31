a = [1,2,3,3,4,4,6,6,6]


def findMeanIndex(nums, key):
    l = len(nums)
    x, y = 0, l-1

    last = 0
    while y>x:
        print(x, y)
        p = int((x+y)/2)+1
        if nums[p]==key:
            last = p
        if nums[p] <= key:
            x = p
        else:
            y = p

    first = l-1
    while y>x:
        print(x, y)
        p = int((x+y)/2)
        if nums[p] == key:
            first = p
        if nums[p]>=key:
            y = p
        else:
            x = p


    return last, first


print(findMeanIndex(a, 6))



