import math
class Solution(object):
    def constructRectangle(self, area):
        mid = int(math.sqrt(area))
        while mid > 0:
            if area % mid == 0:
                return [int(area / mid), int(mid)]
            mid -= 1

#注意： 向下找比向上找要快

x = Solution()
print(x.constructRectangle(2))
