class Solution:
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        l = len(nums)
        if l == 1:
            return 1 if nums[0] == 1 else 0

        maxLength = 0
        startIndex = -1
        endIndex = -1
        for i, num in enumerate(nums):
            if num == 1 and startIndex <= endIndex:
                # print("start: ", i)
                startIndex = i
            if num != 1 and startIndex > endIndex:
                endIndex = i - 1
                # print("end: ", endIndex)
                currentLength = endIndex - startIndex + 1
                if currentLength > maxLength:
                    maxLength = currentLength

        if nums[l-1] == 1 and startIndex>endIndex:
            currentLength =  l-startIndex
            if currentLength > maxLength:
                maxLength = currentLength
        return maxLength

testNums = [1]
x = Solution()
print(x.findMaxConsecutiveOnes(testNums))

