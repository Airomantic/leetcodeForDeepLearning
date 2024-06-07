import sys
from typing import List

def stringToIntegerList(line):
    return list(map(int, line.split()))

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        # 遍历每一行
        for row in matrix:
            # 检查当前行是否包含目标值
            if target in row:
                return True
        return False

def Entermain():
    def readlines():
        for lines in sys.stdin:
            yield lines.strip('\n')

    lines = readlines()
    # 读取第一行，获取矩阵的行数和列数
    m, n = map(int, next(lines).split())
    # 读取下一行，获取目标值
    target = int(next(lines))
    # 读取剩余的行，构建矩阵
    matrix = [stringToIntegerList(next(lines)) for _ in range(m)]
    # 创建Solution对象并调用searchMatrix方法
    result = Solution().searchMatrix(matrix, target)
    # 打印结果
    out = 'true' if result else 'false'
    print(out)

if __name__ == '__main__':
    Entermain()