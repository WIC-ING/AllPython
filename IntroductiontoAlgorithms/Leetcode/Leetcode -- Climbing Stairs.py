class Solution:
    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0 or n==1:
            return 1

        n0, n1 = 1, 1
        for i in range(n-1):
            n1, n0 = n1+n0, n1

        return n1

x = Solution()
print(x.climbStairs(35))

