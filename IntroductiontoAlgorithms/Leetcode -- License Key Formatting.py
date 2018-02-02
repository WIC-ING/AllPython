class Solution:
    def licenseKeyFormatting(self, S, K):
        """
        :type S: str
        :type K: int
        :rtype: str
        """
        allChr =  ''.join(S.split('-')).upper()
        l = len(allChr); key = l%K;
        s = allChr[:key]
        print(key)
        if l < K:
            return allChr
        else:
            while key<=l:
                s = (s + allChr[key:key+K]) if key==l or key==0 else s + '-' + allChr[key:key+K]
                key += K
            return s


x = Solution()
S = "5F3Z-2e-9-w"; K = 1
print(x.licenseKeyFormatting(S, K))
