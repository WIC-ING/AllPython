class Solution:
    def maxAreaOfIsland(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        m, n = len(grid), len(grid[0])

        def DFS(i, j):
            if 0 <= i < m and 0 <= j < n and grid[i][j]:
                grid[i][j] = 0
                return 1 + DFS(i - 1, j) + DFS(i + 1, j) + DFS(i, j - 1) + DFS(i, j + 1)
            return 0

        areas = [DFS(i, j) for i in range(m) for j in range(n)]
        return max(areas)
