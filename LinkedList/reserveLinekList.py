import json
import sys
import io
from typing import Optional # def reverseList(self, head:Optional(ListNode)) -> Optional(ListNode): equivalent: def reverseList(self, head: ListNode | None) -> ListNode | None:

class ListNode:
    def __init__(self, val = 0, next = None) -> None:
        self.val = val
        self.next = next

class Solution:
    def reverseList_iteration(self, head:ListNode) -> ListNode:  # head:ListNode also write head = ListNode
        prev = None
        curr = head
        while curr:
            next = curr.next # preserve pointer
            curr.next = prev  # "cycle" reverse
            
            prev = curr # previous None pointer move next
            curr = next # current None pointer move next

        return prev

class Solution_reverseKGroup:
    def reverse(slef, head: ListNode, tail: ListNode):
        prev  = tail.next # like last "prev = None"
        curr = head
        while prev != tail:
            next = curr.next
            curr.next = prev

            prev = curr
            curr = next
        return tail, head

    def reverseKGroup(self, head: ListNode,  k: int) -> ListNode:
        hair = ListNode(0)
        hair.next = head # dummy pointer point to head None
        pre = hair # temporary preserve hair, now it not already dummy point

        while head:
            tail = pre
            for i in range(k):
                tail = tail.next
                if not tail:
                    return hair.next
            
            nex = tail.next # head None to cycle, meanwhile temporary preserve k-th tail pointer
            head, tail = self.reverse(head, tail) # change paramters position, continue next k array calculate

            # Reconnect the sub LinkedList
            pre.next = head # new head which is the original k-th tail node
            tail.next = nex # create new tail pointer

            pre = tail # new tail "node" use to travesal node cycle next k array
            head = tail.next # new tail "pointer" as head pointer
        return hair.next
 

def listNodeToString(node):
    if not node:
        return "[]"
    
    result = ""
    while node:
        result += str(node.val) + ","
        node = node.next
    return "[" + result[:-1] + "]"  # result[:-1] is last character  ","

def stringToIntegerList(input):
    return json.loads(input)  # notice: have s


def stringToListNode(input):
    numbers = stringToIntegerList(input)

    dummyRoot  = ListNode(0) # temporary save header ponter
    ptr = dummyRoot
    for number in numbers:
        ptr.next = ListNode(number)
        ptr = ptr.next

    ptr = dummyRoot.next # Notice: we're passing the head pointer
    return ptr

def main():
    def readlines():
        for line in io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8'):
            yield line.strip("\n")

    lines = readlines()
    while True:
        try:
            line = next(lines)
            head = stringToListNode(line)
            # result = Solution().reverseList_iteration(head)
            line2 = next(lines)
            k = int(line2)
            result = Solution_reverseKGroup().reverseKGroup(head, k)
            output = listNodeToString(result)
            print(output)
        
        except StopIteration:
            break

if __name__ == "__main__":
    main()