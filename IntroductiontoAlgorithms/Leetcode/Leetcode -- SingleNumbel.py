#Hash表的应用
class Solution:
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        numsDict = {}
        for num in nums:
            if num in numsDict:
                numsDict.pop(num)
            else:
                numsDict[num] = 1

        return numsDict.popitem()[0]