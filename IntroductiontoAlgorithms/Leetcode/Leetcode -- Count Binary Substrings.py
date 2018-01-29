class Solution:
    def countBinarySubstrings(self, s):
        """
        :type s: str
        :rtype: int
        """
        flagChar = s[0]
        numOfChar = 0
        countList = []
        length = len(s)
        for i in range(length):
            c = s[i]
            if c == flagChar:
                numOfChar += 1
            else:
                countList.append(numOfChar)
                numOfChar = 1
                flagChar = c

            if i == length - 1:
                countList.append(numOfChar)

        # print(countList)
        result = 0
        for i in range(len(countList) - 1):
            result += min(countList[i], countList[i + 1])
        return result


