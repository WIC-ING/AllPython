nums = [1, 3, 5, 5, 5, 7, 9]


def binarySearch(nums, n, dir='right'):
    l = len(nums)
    left, right = 0, l - 1

    # 寻找右边界
    if dir == 'right':
        while left <= right:
            key = int((left + right) / 2)

            if nums[key] <= n:
                left += 1
            else:
                right -= 1
        return right
    # 寻找左边界
    elif dir == 'left':
        while left <= right:
            key = int((left + right) / 2)

            if nums[key] < n:
                left += 1
            else:
                right -= 1
        return left


print((binarySearch(nums, 5, 'left') + binarySearch(nums, 5, 'right')) / 2)

