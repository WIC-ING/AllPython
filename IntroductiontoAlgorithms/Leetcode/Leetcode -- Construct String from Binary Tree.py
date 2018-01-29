class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution:
    def tree2str(self, t):
        """
        :type t: TreeNode
        :rtype: str
        """
        if t==None:
            return ''

        if t!= None:
            if t.left==None and t.right==None:
                return str(t.val)

            if t.left!=None and t.right!=None:
                return str(t.val)+'('+self.tree2str(t.left)+')'+'('+self.tree2str(t.right)+')'

            if t.left==None and t.right!=None:
                return str(t.val)+'()'+'('+self.tree2str(t.right)+')'

            if t.left!=None and t.right==None:
                return str(t.val)+'('+self.tree2str(t.left)+')'


node = TreeNode(1)
node.left = TreeNode(2)
node.left.left = TreeNode(4)
node.right = TreeNode(3)

x = Solution()
print(x.tree2str(node))