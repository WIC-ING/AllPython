class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None



def TraverseTree(t):
    if t != None:
        print(t.val, end=' ')
    if t.left != None:
        TraverseTree(t.left)
    if t.right != None:
        TraverseTree(t.right)

def merge(t, t1, t2):
    if t1==None:
        return t2
    if t2==None:
        return t1
    t.val = t1.val + t2.val

    if t1.left!=None or t2.left!=None:
        t.left = TreeNode(0)
        t.left = merge(t.left, t1.left, t2.left)
    if t1.right!=None or t2.right!=None:
        t.right = TreeNode(0)
        t.right = merge(t.right, t1.right, t2.right)
    return t




class Solution(object):
    def mergeTree(self, t1, t2):
        t = TreeNode(0)
        t = merge(t, t1, t2)
        return t


t1 = TreeNode(3)
t1.left  = TreeNode(4)
t1.right = TreeNode(5)
t1.left.left  = TreeNode(1)
t1.left.right = TreeNode(2)

t2 = TreeNode(4)
t2.left = TreeNode(1)
t2.right = TreeNode(2)
# t2.left.right = TreeNode(4)
# t2.right.right = TreeNode(7)

x = Solution()
TraverseTree(x.mergeTree(t1, t2))

