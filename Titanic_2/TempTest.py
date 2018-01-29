class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """


x=1
y=4

s = Solution()
print(s.hammingDistance(x,y))


# import pandas as pd
# import numpy as np
#
# data = {'a':[1,2,3], 'b':[2,3,3]}
#
# df = pd.DataFrame(data)
#
# df = df.rename(columns={'b':'c'})
#
# cols = list(df)
#
# print(type(cols))
#
# cols.insert(0, cols.pop(cols.index('c')))
#
# df = df.ix[:, cols]
#
# print(df)
