class treeNode():
    def __init__(self, val):
        self.val   = val
        self.left  = None
        self.right = None


def BFS(root):
    nodeList = [root]
    for node  in nodeList:
        if node.left!=None:
            nodeList.append(node.left)
        if node.right!=None:
            nodeList.append(node.right)
    return nodeList


testTree = treeNode(1)
testTree.left = treeNode(2)
testTree.left.left  = treeNode(4)
testTree.left.right = treeNode(5)

testTree.right = treeNode(3)
testTree.right.left = treeNode(6)
testTree.right.right = treeNode(7)


BFS(testTree)
