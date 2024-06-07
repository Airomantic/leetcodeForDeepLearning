import io
import json
import sys
from typing import List


def stringToIntegerList(line):
    return json.loads(line)

class Solution:
    def WaterMaxContainer(self,height:List[int]):
        i,j=0,len(height)-1
        maxArea=0
        while i<j:
            if height[i]>height[j]:
                h,w=height[j],j-i
                j-=1
            else:
                h, w = height[i], j - i
                i+=1
            maxArea=max(maxArea,h*w)
        return maxArea

def Entermain():
    def readlines():
        for lines in io.TextIOWrapper(sys.stdin.buffer,encoding='utf-8'):
            yield lines.strip('\n')

    lines=readlines()
    while True:
        try:
            line=next(lines)
            LN=stringToIntegerList(line)
            result=Solution().WaterMaxContainer(LN)
            out=str(result)
            print(out)
        except StopIteration:
            break

if __name__ == '__main__':
    Entermain()