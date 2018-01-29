primeList = [2,3,5,7,11,13,17,19]
class Solution:
    def countPrimeSetBits(self, L, R):
        """
        :type L: int
        :type R: int
        :rtype: int
        """
        return sum(bin(x).count('1') in primeList for x in range(L, R+1))

L = 244
R = 269

x = Solution()
print(x.countPrimeSetBits(L,R))
