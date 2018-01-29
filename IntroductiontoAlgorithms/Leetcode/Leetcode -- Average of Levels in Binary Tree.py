# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

#广度优先遍历
def BFS(nodeList):
    if len(nodeList) == 0:
        return
    queue = []
    nodeVals = []
    for node in nodeList:
        nodeVals.append(node.val)
        # print(node.val)
        if node.left != None:
            queue.append(node.left)
        if node.right != None:
            queue.append(node.right)
    return  sum(nodeVals)/len(nodeVals), queue
    # BFS(queue)



class Solution:
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        nodeList = [root]
        resultList = []
        while len(nodeList) != 0:
            aveval, nodeList = BFS(nodeList)
            resultList.append(aveval)
        return resultList






testNode = TreeNode(3)
testNode.left = TreeNode(9)
testNode.right = TreeNode(20)
testNode.right.left = TreeNode(15)
testNode.right.right = TreeNode(7)

x = Solution()
print(x.averageOfLevels(testNode))
