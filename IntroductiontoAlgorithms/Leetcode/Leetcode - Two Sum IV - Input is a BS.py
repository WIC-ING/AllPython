class Solution:
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        targetNum = set()

        def tarvelTree(root, k):
            # print(targetNum)
            if root != None:
                # print('root->val', root.val)
                if root.val in targetNum:
                    return True
                if root.val != None:
                    targetNum.add(k - root.val)
                if tarvelTree(root.left, k):
                    return True
                elif tarvelTree(root.right, k):
                    return True
            return False

        return tarvelTree(root, k)


class TreeNode():
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

node = TreeNode(2)
node.left = TreeNode(0)
node.right = TreeNode(3)
node.left.left = TreeNode(-4)
node.left.right = TreeNode(1)
# node.right.right = TreeNode(7)


x = Solution()
print(x.findTarget(node, -1))

