class Solution:
    def nextGreaterElement(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        #时间复杂度为O（n）的解法：
        #
        #1.大体思路：首先对nums2，进行处理，得到nums2中每一个元素为key值，右侧第一个较大数字为value值的dictionary
        #然后通过对nums1中元素逐个循环，得到相应的答案
        #
        #2.原则：我原来的思路时时间复杂度为O（n^2的解法）
        # 其问题在于：既然在处理nums1中的第一个元素的时候已经对nums2进行了一次遍历，就应该把这次遍历的到的信息利用起来，
        # 而不是循环的多次遍历nums2
        #
        #3.须记住的基础知识： a: dict.get()
        #                  b: list.pop()
        #
        #4.所暴露的只是漏洞： a: 对python的一些基础函数调用不熟悉
        #                  b: 最主要是对堆栈这种数据结构理解不深刻
        #


        d = {}
        st = []
        ans = []

        for x in nums2:
            while len(st) and st[-1] < x:
                d[st.pop()] = x
            st.append(x)

        for x in nums1:
            ans.append(d.get(x, -1))

        return ans

        # outNums = []
        #
        # for key in nums1:
        #     Find = False
        #     Exist = False
        #     for data in nums2:
        #         print(key, ':', Find, Exist)
        #         if data == key:
        #             Find = True
        #         if Find and data > key:
        #             outNums.append(data)
        #             Exist = True
        #             break
        #     if Exist == False:
        #         outNums.append(-1)
        #     print(key, ":", outNums)
        # return outNums

nums1 = [4,1,2]
nums2 = [1,3,4,2]

x = Solution()
print(x.nextGreaterElement(nums1, nums2))
