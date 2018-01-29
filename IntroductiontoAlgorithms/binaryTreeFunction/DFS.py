class treeNode():
    def __init__(self, val):
        self.val   = val
        self.left  = None
        self.right = None

# 递归前序遍历
def DFS_pre(root):
    if root==None:
        return
    print(root.val)
    DFS_pre(root.left)
    DFS_pre(root.right)

# 非递归前序遍历
def DFS_preNoRecursive(root):
    nodeList = []
    p = root
    while(p!=None) or nodeList:
        while p!=None:
            print(p.val)
            nodeList.append(p)
            p = p.left
        if nodeList:
            p = nodeList[-1]
            nodeList.pop()
            p=p.right

# 递归中序遍历
def DFS_med(root):
    if root==None:
        return
    DFS_med(root.left)
    print(root.val)
    DFS_med(root.right)

# 非递归中序遍历
def DFS_medNoRecursive(root):
    nodeList = []
    p = root
    while p!=None or nodeList:
        while p!=None:
            nodeList.append(p)
            p = p.left
        if nodeList:
            p = nodeList[-1]
            print(p.val)
            nodeList.pop()
            p = p.right

# 递归后序遍历
def DFS_back(root):
    if root==None:
        return
    DFS_back(root.left)
    DFS_back(root.right)
    print(root.val)

# 非递归后序遍历
def DFS_backNoRecursive(root):
    nodeList = []
    p = root
    lastP = None
    nodeList.append(root)

    while nodeList:
        p = nodeList[-1]
        if (p.left==None and p.right==None) or \
                (lastP!=None and (lastP==p.left or lastP==p.right)):
            print(p.val)
            nodeList.pop()
            lastP = p
        else:
            if p.right!=None:
                nodeList.append(p.right)
            if p.left!=None:
                nodeList.append(p.left)




testTree = treeNode(1)
testTree.left = treeNode(2)
testTree.left.left  = treeNode(4)
testTree.left.right = treeNode(5)

testTree.right = treeNode(3)
testTree.right.left = treeNode(6)
testTree.right.right = treeNode(7)


# DFS_pre(testTree)
# DFS_med(testTree)
# DFS_back(testTree)

# DFS_preNoRecursive(testTree)
# DFS_medNoRecursive(testTree)
DFS_backNoRecursive(testTree)


