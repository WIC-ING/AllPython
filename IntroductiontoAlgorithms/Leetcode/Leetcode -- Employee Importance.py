
# Employee info
class Employee:
    def __init__(self, id, importance, subordinates):
        # It's the unique id of each node.
        # unique id of this employee
        self.id = id
        # the importance value of this employee
        self.importance = importance
        # the id of direct subordinates
        self.subordinates = subordinates


class Solution:
    def getImportance(self, employees, id):
        """
        :type employees: Employee
        :type id: int
        :rtype: int
        """
        self.tempDict = {}
        self.getDict(employees)
        return self.calcImportance(id)

    def getDict(self, employees):
        for employee in employees:
            self.tempDict[employee.id] = employee

    def calcImportance(self, key):
        result = self.tempDict[key].importance

        for employeeId in self.tempDict[key].subordinates:
            result += self.calcImportance(employeeId)
        return result

# class Solution:
#     def getImportance(self, employees, id):
#         """
#         :type employees: Employee
#         :type id: int
#         :rtype: int
#         """
#
#         ImportanceSum = 0
#         for employee in employees:
#             if employee.id == id:
#                 ImportanceSum += employee.importance
#                 if employee.subordinates == []:
#                     return employee.importance
#                 for subId in employee.subordinates:
#                     ImportanceSum += self.getImportance(employees, subId)
#         return ImportanceSum

emp1 = Employee(1, 5, [2,3])
emp2 = Employee(2, 3, [])
emp3 = Employee(3, 3, [])

testEmployees = [emp1, emp2, emp3]
testId = 1

x = Solution()
print(x.getImportance(testEmployees, testId))
