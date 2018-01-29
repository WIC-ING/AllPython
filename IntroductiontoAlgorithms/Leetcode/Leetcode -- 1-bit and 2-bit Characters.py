class Solution:
    def isOneBitCharacter(self, bits):
        """
        :type bits: List[int]
        :rtype: bool
        """
        l = len(bits)
        if l == 1:
            return True
        elif l == 2:
            return True if bits[0] == 0 else False
        else:
            i = 0
            while (i < l):
                if i == l - 1:
                    return True
                elif bits[i] == 1:
                    i += 2
                else:
                    i += 1
            return False

l = [1,1,1,0]
x = Solution()
print(x.isOneBitCharacter(l))