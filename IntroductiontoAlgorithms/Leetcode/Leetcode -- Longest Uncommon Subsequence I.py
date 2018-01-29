class Solution:
    def findLUSlength(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: int
        """
        m, n = len(a), len(b)
        if m != n:
            return max(m, n)

        minIndex, maxIndex = float('inf'), float('-inf')
        for i in range(m):
            if a[i] != b[i]:
                if i < minIndex:
                    minIndex = i
                if i > maxIndex:
                    maxIndex = i
        if minIndex <= maxIndex:
            return maxIndex - minIndex + 1
        else:
            return -1

a = 'aba'
b = 'cdc'

x = Solution()
print(x.findLUSlength(a,b))

