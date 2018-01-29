import heapq

class Solution:
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # -------------------------------
        # ------------完美解法------------
        # -------------------------------
        a, b = heapq.nlargest(3, nums), heapq.nsmallest(2, nums)

        return max(a[0] * a[1] * a[2], b[1] * b[0] * a[0])

        #-------------------------------
        #------------自己解法-----------
        #-------------------------------
        # l = len(nums)
        # if l == 3:
        #     return nums[0] * nums[1] * nums[2]
        # sort_index = sorted(list(range(l)), key=lambda k: abs(nums[k]), reverse=True)
        #
        # temp = [nums[sort_index[0]], nums[sort_index[1]], nums[sort_index[2]]]
        # print(temp)
        #
        # negNums = 0
        # negIndex = 0
        # for i in range(3):
        #     if temp[i] < 0:
        #         negIndex = i
        #         negNums += 1
        # print(negIndex)
        #
        # if negNums == 0 or negNums == 2:
        #     return temp[0] * temp[1] * temp[2]
        # else:
        #     maxPos = 0
        #     maxNeg = 0
        #     for j in sort_index[3:]:
        #         if nums[j] > 0:
        #             maxPos = nums[j]
        #             break
        #     for j in sort_index[3:]:
        #         if nums[j] < 0:
        #             maxNeg = nums[j]
        #             break
        #
        #     temp1 = temp.copy()
        #     temp1.pop(negIndex)
        #     data1 = temp1[0] * temp1[1] * maxPos
        #     # print('data1: ', data1, 'maxPos: ', maxPos)
        #
        #     temp2 = [1,2,0]; temp2.remove(negIndex)
        #     data2 = temp[min(temp2)] * temp[negIndex] * maxNeg
        #     # print('data2: ', data2, 'maxNeg: ', maxNeg)
        #     return max(data1, data2)

l = [903,606,48,-474,313,-672,872,-833,899,-629,558,-368,231,621,716,-41,-418,204,-1,883,431,810,452,-801,19,978,542,930,85,544,-784,-346,923,224,-533,-473,499,-439,-925,171,-53,247,373,898,700,406,-328,-468,95,-110,-102,-719,-983,776,412,-317,606,33,-584,-261,761,-351,-300,825,224,382,-410,335,187,880,-762,503,289,-690,117,-742,713,280,-781,447,227,-579,-845,-526,-403,-714,-154,960,-677,805,230,591,442,-458,-905,832,-285,511,536,-86]
x = Solution()
print(x.maximumProduct(l))
