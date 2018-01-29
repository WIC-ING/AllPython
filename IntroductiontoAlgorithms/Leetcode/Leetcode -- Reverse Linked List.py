# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def showAll(node):
    print('showAll:')
    while(node!=None):
        print(node.val)
        node = node.next


class Solution:
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        lastNode = None
        while head != None:
            nextNode = head.next
            head.next = lastNode
            lastNode = head
            head = nextNode

        return lastNode


l = ListNode(1)
l.next = ListNode(2)
l.next.next = ListNode(3)
l.next.next.next = ListNode(4)
l.next.next.next.next = ListNode(5)

x = Solution()
# print(x.reverseList(l))
showAll(x.reverseList(l))