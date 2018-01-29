class Solution:
    def toHex(self, num):
        """
        :type num: int
        :rtype: str
        """
        bin2hex = {'0000': '0', '0001': '1', '0010': '2', '0011': '3', '0100': '4', '0101': '5', '0110': '6',
                   '0111': '7',
                   '1000': '8', '1001': '9', '1010': 'a', '1011': 'b', '1100': 'c', '1101': 'd', '1110': 'e',
                   '1111': 'f'}
        binNum = ''
        if num >= 0:
            binNum = bin(num)[2:]
            binNum = '0' * (32 - len(binNum)) + binNum
        else:
            # binNum = ''.join('1' if c=='0' else '0' for c in bin(-num)[2:])
            binNum = bin(-num)[2:]
            binNum = '0' * (32 - len(binNum)) + binNum
            binNum = bin(int(''.join('1' if c == '0' else '0' for c in binNum), 2) + 1)[2:]



        s = ''
        for i in range(8):
            s = s + bin2hex[binNum[(4 * i):(4 * i + 4)]]
        start = 7
        for i in range(8):
            if s[i] != '0':
                start = i
                break
        return s[start:]

x = Solution()
print(x.toHex(26))
