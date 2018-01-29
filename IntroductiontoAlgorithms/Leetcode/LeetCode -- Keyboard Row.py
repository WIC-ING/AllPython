def getRowth(c):
    if c in 'qwertyuiop' or c in 'QWERTYUIOP':
        return 1
    elif c in 'asdfghjkl' or c in 'ASDFGHJKL':
        return 2
    else:
        return 3

def is_same_row(w):
    rowArray = [getRowth(c) for c in w]
    key = rowArray[0]
    return all([row == key for row in rowArray])

class Solution:
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        return list(filter(is_same_row, words))

# print(getRowth('H'))

words = ["Hello", "Alaska", "Dad", "Peace"]

x = Solution()
print(x.findWords(words))

print(is_same_row('Hello'))