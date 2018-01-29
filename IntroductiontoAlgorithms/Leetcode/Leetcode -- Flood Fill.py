class Solution:
    def floodFill(self, image, sr, sc, newColor):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type newColor: int
        :rtype: List[List[int]]
        """
        m, n = len(image), len(image[0])

        if (m == 0 and n == 0) or (newColor==image[sr][sc]):
            return image



        temp = image[sr][sc]
        image[sr][sc] = newColor

        if sr > 0 and image[sr - 1][sc] == temp:
            self.floodFill(image, sr - 1, sc, newColor)
        if sr < m - 1 and image[sr + 1][sc] == temp:
            self.floodFill(image, sr + 1, sc, newColor)
        if sc > 0 and image[sr][sc - 1] == temp:
            self.floodFill(image, sr, sc - 1, newColor)
        if sc < n - 1 and image[sr][sc + 1] == temp:
            self.floodFill(image, sr, sc + 1, newColor)

        return image

testList = [[0,0,0],[0,1,1]]

x = Solution()
print(x.floodFill(testList, 1, 1, 1))

