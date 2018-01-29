class Solution:
    def longestWord(self, words):
        """
        :type words: List[str]
        :rtype: str
        """
        rs = ''
        maxL = 0

        wordset = set(words)
        for word in words:
            l = len(word)
            # print(word, list(word[:k] for k in range(1,l)))
            if all(word[:k] in wordset for k in range(1,l)):
                print(word)
                if l > maxL or (l == maxL and word < rs):
                    rs = word
                    maxL = l

        return rs

s = ["yo","ew","fc","zrc","yodn","fcm","qm","qmo","fcmz","z","ewq","yod","ewqz","y"]
x = Solution()
print('answer: ', x.longestWord(s))

