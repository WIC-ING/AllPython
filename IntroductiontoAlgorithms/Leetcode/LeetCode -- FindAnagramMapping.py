class Solution:
    def anagramMappings(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        B_index = {}
        P = []
        j = 0
        for i, a in enumerate(A):
            if a in B_index:
                P.append(B_index[a])
            else:
                while(a != B[j]):
                    B_index[B[j]] = j
                    j+=1
                P.append(j)
        return P

A = [12, 28, 46, 32, 50]
B = [50, 12, 32, 46, 28]

x = Solution()
print(x.anagramMappings(A,B))


#A little change of it
#Second change of it