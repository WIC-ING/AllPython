def oneLineNums(nums,m,n):
    data = []
    for i in range(m):
        for j in range(n):
            data.append(nums[i][j])
    return  data

# print(oneLineNums(nums))

class Solution:
    def matrixReshape(self, nums, r, c):
        """
        :type nums: List[List[int]]
        :type r: int
        :type c: int
        :rtype: List[List[int]]
        """
        m, n = len(nums), len(nums[0])
        if m*n != r*c:
            return nums
        key=0
        outNums = []

        oneLineData = oneLineNums(nums,m,n)

        for i in range(r):
            outNums.append([])
            for j in range(c):
                outNums[i].append(oneLineData[key])
                key+=1
        return outNums

nums = [[1,2],[3,4]]

r = 1; c = 4

x = Solution()

print(x.matrixReshape(nums,r,c))
