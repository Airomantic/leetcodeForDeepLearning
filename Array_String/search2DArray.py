import sys
import io
import json
from typing import List
# input:
# [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
# 3

def stringToIntegerList(line):
    return json.loads(line)

class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        if not matrix or not matrix[0]:
            return False
        
        rows, cols = len(matrix), len(matrix[0])
        row, col = 0, cols - 1 # cols - 1 : Sequence  retrieval from large to small
        while row < rows and col >= 0 :
            if target ==  matrix[row][col]:
                return True
            elif target < matrix[row][col]:
                col -= 1 # retrieval in this line which is numbe "row"
            else:
                row += 1 # key point: similar to binary search

        return False

def Entermain():
    def readlines():
        for lines in sys.stdin:
            yield lines.strip('\n')

    lines = readlines()
    # 读取第一行，获取矩阵的行数和列数
    matrix_str = next(lines)
    matrix = json.loads(matrix_str)  # 使用 json 直接解析整个矩阵
    # 读取下一行，获取目标值
    # 读取目标值
    target_str = next(lines)
    target = int(target_str)

    # 创建Solution对象并调用searchMatrix方法
    result = Solution().searchMatrix(matrix, target)
    # 打印结果
    out = 'true' if result else 'false'
    print(out)

if __name__ == '__main__':
    Entermain()