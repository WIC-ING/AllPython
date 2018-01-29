def distance(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2


class Solution:
    def numberOfBoomerangs(self, points):
        """
        :type points: List[List[int]]
        :rtype: int
        """
        m = len(points)
        result = 0
        for i in range(m):
            distDict = {}
            for j in range(m):
                dist = distance(points[i], points[j])
                distDict[dist] = 1 + distDict.get(dist, 0)

            for key in set(distDict):
                result += distDict[key] * (distDict[key] - 1)

        return result

