class Solution:
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        return sum(y - x for x, y in zip(prices[:-1], prices[1:]) if y > x)

#         l = len(prices)
#         priceDiff = []
#         for i in range(l-1):
#             priceDiff.append (1 if prices[i+1]>=prices[i] else 0)

#         buyIndex  = -1
#         sellIndex = -1
#         totalProfit = 0
#         for i in range(l-1):
#             if priceDiff[i] == 0 and buyIndex>sellIndex:
#                 sellIndex = i
#                 totalProfit += prices[sellIndex] - prices[buyIndex]
#                 buyIndex = i
#             elif priceDiff[i]==1 and buyIndex==sellIndex:
#                 buyIndex = i

#         if buyIndex > sellIndex:
#             totalProfit += prices[-1]-prices[buyIndex]

#         return totalProfit