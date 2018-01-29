# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def DFS(node):
    if node==None:
        return
    print(node.val)
    DFS(node.left)
    DFS(node.right)

def BFS(node):
    if node==None:
        return node

    nl = [node]
    for n in nl:
        if n.left!=None:
            nl.append(n.left)
        if n.right!=None:
            nl.append(n.right)
    return [node.val for node in nl]


def DFS_dealData(node, allNum):
    if node==None:
        return
    node.val += sum( list(filter( lambda x: x>node.val, allNum)) )
    DFS_dealData(node.left,allNum)
    DFS_dealData(node.right, allNum)


Sum = 0
class Solution:

    def convertBST(self, root):
        global Sum
        if root==None:
            return None
        self.convertBST(root.right)
        temp = root.val
        root.val += Sum
        Sum += temp
        self.convertBST(root.left)
        return root

# list(filter(lambda x: x>2, l))
node = TreeNode(2)
node.left = TreeNode(1)
node.right = TreeNode(3)

# node.left.left = TreeNode(-4)
# node.left.right = TreeNode(1)


# DFP(node)
# print(BFP(node))

x = Solution()
print(BFS(x.convertBST(node)))
