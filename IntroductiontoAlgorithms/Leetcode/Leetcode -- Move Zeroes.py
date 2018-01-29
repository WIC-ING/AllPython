class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        l = len(nums)
        noZeroIndex = []
        for i in range(l):
            if nums[i] != 0:
                noZeroIndex.append(i)

        numOfNoZero = len(noZeroIndex)
        index = 0
        for i in range(l):
            if i < numOfNoZero:
                nums[i] = nums[noZeroIndex[index]]
                index += 1
            else:
                nums[i] = 0


