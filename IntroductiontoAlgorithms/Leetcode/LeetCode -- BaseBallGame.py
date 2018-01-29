"""
Given a list of strings, each string can be one of the 4 following types:

Integer (one round's score): Directly represents the number of points you get in this round.
"+" (one round's score): Represents that the points you get in this round are the sum of the last two valid round's points.
"D" (one round's score): Represents that the points you get in this round are the doubled data of the last valid round's points.
"C" (an operation, which isn't a round's score): Represents the last valid round's points you get were invalid and should be removed.
"""

s1 = ["5","2","C","D","+"]
s2 = ["5","-2","4","C","D","9","+","+"]
s3 = ['2','4','+', 'C', 'D']
s4 = ["5","-2"]
class Solution:
    def calPoints(self, ops):
        """
        :type ops: List[str]
        :rtype: int
        """
        last_valid_points = []

        for x in ops:
            if x == 'C':
                last_valid_points.pop()
            elif x == '+':
                last_valid_points.append(last_valid_points[-1] + last_valid_points[-2])
            elif x == 'D':
                last_valid_points.append(last_valid_points[-1]*2)
            else:
                last_valid_points.append(int(x))

        return sum(last_valid_points)

x = Solution()
print(x.calPoints(s1))
