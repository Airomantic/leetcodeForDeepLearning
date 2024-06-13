import heapq

class MaxHeap:
    def __init__(self) -> None:
        self.heap = []

    def push(self, value):
        # take a negative values and insert it into the heap
        heapq.heappush(self.heap, -value)

    def pop(self):
        # Pop the maximum value from the heap and take a negative value restore the origin value
        return -heapq.heappop(self.heap)

    def peek(self):
        # View the maximum value in the heap without popping
        return -self.heap[0]

    def __len__(self):
        return len(self.heap)



if __name__ == "__main__":

    max_heap = MaxHeap()

    max_heap.push(10)
    max_heap.push(20)
    max_heap.push(1)
    max_heap.push(5)

    print("max heap = ", max_heap.pop())
    print("max heap = ", max_heap.pop())
    print("max heap = ", max_heap.pop())
    print("max heap = ", max_heap.pop())