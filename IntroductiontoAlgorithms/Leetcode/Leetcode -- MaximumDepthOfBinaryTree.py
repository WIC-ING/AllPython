class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root == None:
            return 0

        leftDepth = self.maxDepth(root.left)
        rightDepth = self.maxDepth(root.right)

        return leftDepth + 1 if leftDepth >= rightDepth else rightDepth + 1