import heapq

def a_star(maze, start, goal):
    def heuristic(a, b):  # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def neighbors(node):
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # binary array: right, up, down, left
        result = []
        for d in dirs:
            neighbor = (node[0] + d[0], node[1] + d[1])
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0:
                result.append(neighbor)
        return result

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start)) # open_set is the name of the smallest heap, which is a list of triples about (f_score, g_score, node)
    came_from = {}
    g_score = {start: 0} # coordinate (x, y) can as key, it is corresponding value is a numerical value, which represents the "const"
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[2] # The node with lowest f_score are processed each time

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for neighbor in neighbors(current):
            tentative_g_score = g_score[current] + 1
            # neighbor not in g_score: if neighbor can passable and check whether the neighbor node has not been accessed
            # tentative_g_score < g_score[neighbor]: the direction chosen causes the total cost to be smaller than before
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # update path info
                came_from[neighbor] = current # came_from: backtracking path, record the previous coordinate point
                g_score[neighbor] = tentative_g_score                                 # the actual cost from the starting point to the neighbor node
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)     # Estimated total cost from the starting point to the target node (actual cost + estimated cost)
                heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

    return []

# 示例使用
maze = [
    [0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

start = (0, 0)
goal = (4, 4)

path = a_star(maze, start, goal)
print("Path found by A*:", path)
