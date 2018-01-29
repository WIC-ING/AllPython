# Definition for a binary tree node.
def travelBinarytree(t):
    print(t.val)
    if t.left != None:
        travelBinarytree(t.left)
    if t.right != None:
        travelBinarytree(t.right)


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def trimBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: TreeNode
        """
        print('trimTree: ', root.val)
        if root.val > R:
            if root.left != None:
                root = self.trimBST(root.left, L, R)
            else:
                root = None
        elif root.val < L:
            if root.right != None:
                root = self.trimBST(root.right, L, R)
            else:
                root = None
        else:
            if root.left != None:
                root.left = self.trimBST(root.left, L, R)

            if root.right != None:
                root.right = self.trimBST(root.right, L, R)
        return root

TestNode = TreeNode(3)
TestNode.left =TreeNode(1)
TestNode.right =TreeNode(4)
TestNode.left.right = TreeNode(2)

x = Solution()
root = x.trimBST(TestNode, 3, 4)
print('\n\n')
travelBinarytree(root)



#------------------------------------------
#--------------------标准答案---------------
#------------------------------------------
# class Solution(object):
#     def trimBST(self, root, L, R):
#         def trim(node):
#             if not node:
#                 return None
#             elif node.val > R:
#                 return trim(node.left)
#             elif node.val < L:
#                 return trim(node.right)
#             else:
#                 node.left = trim(node.left)
#                 node.right = trim(node.right)
#                 return node
#
#         return trim(root)
#
#运行时间和运行复杂度都相同