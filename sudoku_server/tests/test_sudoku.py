import unittest
import numpy as np

from sudoku_server.utils.sudoku import SudokuHandler


class SudokuCase(unittest.TestCase):
    def test_hidden(self):
        s = SudokuHandler()
        key = "unc_row_0"
        row = [{1, 4, 3}, {2, 3}, {7, 5, 2, 3}, {1, 4, 2, 3}, {6, 2, 3}, {6, 2, 3}, {7, 2, 5}, 8, 9]
        s.gen_abc_hidden(key, row)
        r = [{1, 4}, {2, 3}, {7, 5}, {1, 4}, {6, 2, 3}, {6, 2, 3}, {7, 5}, 8, 9]
        self.assertEqual(r, row)

    def test_hidden2(self):
        """测试重试当前行，如果不重试则不符合测试结果"""
        s = SudokuHandler()
        key = "unc_row_0"
        row = [{1, 2, 3, 4}, {2, 3, 4}, {1, 4, 5, 6}, {1, 4, 5, 6}, {5, 7}, {7, 6}, {5, 6}, 8, 9]
        s.gen_abc_hidden(key, row)
        r = [{2, 3}, {2, 3}, {1, 4}, {1, 4}, {5, 7}, {7, 6}, {5, 6}, 8, 9]
        self.assertEqual(r, row)

    def test_hidden3(self):
        """测试重试当前行，如果不重试则不符合测试结果"""
        s = SudokuHandler()
        key = "unc_row_0"
        row = [{1, 2, 3, 4, 5}, 9, {4, 5}, {1, 2, 3, 4}, {4, 5}, {7, 6}, {5, 6}, 8, {1, 2, 3, 5}]
        list(s.gen_abc_hidden(key, row))
        r = [{1, 2, 3}, 9, {4, 5}, {1, 2, 3}, {4, 5}, {7, 6}, {5, 6}, 8, {1, 2, 3}]
        self.assertEqual(r, row)

    def test_single_hidden(self):
        s = SudokuHandler()
        key = "unc_row_0"
        row = np.array([{4, 3}, {2, 3}, {7, 5, 2, 3}, {1, 4, 2, 3}, {6, 2, 3}, {6, 2, 3}, {7, 2, 5}, 8, 9])
        s.gen_abc_hidden_single(key, row)
        r = [{4, 3}, {2, 3}, {7, 5, 2, 3}, {1}, {6, 2, 3}, {6, 2, 3}, {7, 2, 5}, 8, 9]
        self.assertEqual(r, row.tolist())

    def test_single_hidden(self):
        s = SudokuHandler()
        key = "unc_row_0"
        row = np.array([{1, 2, 3}, {2, 3}, {4, 2, 3}, {4, 5}, {6, 5}, 7, {4, 6, 5}, 8, 9])
        s.gen_abc_hidden_single(key, row)
        r = [{1}, {2, 3}, {4, 2, 3}, {4, 5}, {6, 5}, 7, {4, 6, 5}, 8, 9]
        self.assertEqual(r, row.tolist())

    def test_single_hidden2(self):
        s = SudokuHandler()
        key = "unc_row_0"
        row = np.array([{6, 5}, {2, 3}, {4, 2, 3}, {4, 5}, {1, 2, 6, 3}, {5, 7}, {4, 7, 5}, 8, 9])
        s.gen_abc_hidden_single(key, row)
        # s.abc_hidden_single(key, row)
        r = [{6}, {2, 3}, {4, 2, 3}, {4, 5}, {1}, {5, 7}, {4, 7, 5}, 8, 9]
        self.assertEqual(r, row.tolist())


if __name__ == '__main__':
    unittest.main()
