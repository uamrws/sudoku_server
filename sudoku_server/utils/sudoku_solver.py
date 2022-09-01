"""
1. 将每一行的结果确定
   a. 当前格为空白处的，值等于「1-9」的集合减去「当前行所有确定的数的集合」
   b. 如果当前格为集合的，且减去「所有其他格的集合的并集」为1的，可以确定当前处的值为差集所得数
2. 将每一宫的结果确定
   确定方法一致
3. 将每一竖的结果确定
   确定方法一致
4. 重复1-3的流程2-3次后，如果校验后未解出答案说明数独需要试数
5. 试数步骤将某一处的集合暂时取值后重复1-4的解题步骤，直到得出结果，存起来，继续试数得出其他结果

数独解法规则：
规则 1 显式唯一候选数法(Naked Single):
    若某行，或列， 或九宫格中的某个空格候选数唯一，那么该候选数就填该 空格。

规则 2 隐式唯一候选数法(Hidden Single):
    若某个候选 数只出现在某行，或列，或九宫格的某个空格中，那么该空 格就填该候选数。

规则 3 显式二数集删减法(Naked Pair):
    若某行，或列， 或九宫格中的某 2 个空格的候选数恰好为 2 个，那么这 2 个 候选数可以从该行，或列，或九宫格的其他空格候选数中 删除。

规则 4 隐式二数集删减法(Hidden Pair):
    若某 2 个候选 数只出现在某行，或列，或九宫格中的某 2 个空格中，那么 这 2 个空格中不同于这 2 个候选数的其他候选数可删除。

规则 5 显式三数集删减法(Naked Triplet):
    若某行，或列， 或九宫格中的某 3 个空格的候选数恰好为 3 个，那么这 3 个 候选数可以从该行，或列，或九宫格的其他空格候选数中删除。

规则 6 隐式三数集删减法(Hidden Triplet):
    若某 3 个候选数只出现在某行，或列，或九宫格中的某 3 个空格中，那 么这 3 个空格中不同于这 3 个候选数的其他候选数可删除。

规则 7 显式四数集删减法(Naked Quad):
    若某行，或列， 或九宫格中的某 4 个空格的候选数恰好为 4 个，那么这 4 个 候选数可以从该行，或列，或九宫格的其他空格候选数中 删除。

规则 8 隐式四数集删减法(Hidden Quad):
    若某 4 个候选 数只出现在某行，或列，或九宫格中的某 4 个空格中，那么 这 4 个空格中不同于这 4 个候选数的其他候选数可删除。

规则 9 区块删减法(Intersection Removel)
    区块对行的影响:在某一区块中，当所有可能出现某个 数字的单元格都位于同一行时，就可以把这个数字从该行的 其他单元格的候选数中删除。
    区块对列的影响:在某一区块中，当所有可能出现某个 数字的单元格都位于同一列时，就可以把这个数字从该列的 其他单元格的候选数中删除。
    行或列对区块的影响:在某一行(列)中，当所有可能出 现某个数字的单元格都位于同一区块中时，就可以把这个数 字从该区块的其他单元格的候选数中删除。

规则 10 矩形对角线删减法(X-wing):
    如果一个数字正好 出现且只出现在某 2 行(列)相同的 2 列(行)上，则这个数字就 可以从这 2 列(行)上其他单元格的候选数中删除。

规则 11 三链数删减法(Swordfish):
    如果某个数字在某 3 列(行)中只出现在相同的 3 行(列)中，则这个数字将从这 3 行(列)上其他的候选数中删除。

规则 12 XY 形态匹配法(XY-wing):
    若 XY 和 YZ 同在一 个区块但不同行(列)中，而 XZ 和 XY 在同一行(列)，但在不 同区块中。那么在 XY 所在区块中与 XY 所在行(列)交集空 格中应该删除候选数 Z，并且在 XZ 所在区块与 YZ 所在行 (列)交集的空格中应该删除候选数 Z。其中，XY、YZ、XZ 分别是三空格的候选数，并且这三空格没有其他候选数。

规则 13 XYZ 形态匹配法(XYZ-wing):
    若某区块某空格 候选数为 XYZ，在该同区块但不同列(行)的某空格候选数为 YZ，且与 XYZ 所在空格同列(行)但不同区块某空格候选数为 XZ，那么 XYZ 所在区块与 XZ 所在列(行)的交集中的空格候 选数不能为 Z。

规则 14 试探法:
    若某个空格的候选数只有 2 个时，进 行试探填写其中一个候选数，若填写成功则该试探成功，若 导致矛盾则另外一个候选数应填该空格。
"""
import copy
import random
from functools import reduce
from operator import ior

import numpy as np


class SudokuHandler(object):

    def __init__(self):
        self.standard_set = set(range(1, 10))
        self.table = None
        self.is_int = lambda x: isinstance(x, int)
        self.is_set = lambda x: isinstance(x, set)
        # 所有结果集
        self.results = []
        self.ONLY_ONE = False

    def build_from_list(self, a_list: list) -> object:
        """从表格导入数据"""
        self.table = np.array(a_list)
        return self

    def _filter(self, item, index, res_set, set_indexes, empty_indexes):
        """
            将每行/列/宫的数据中的数据按以下标准筛选出下标
            1.已经确定的数字（单元格属性为int）
            2.未确定的数字（单元格属性为set）
            3.还未初始化的（单元格为空的）
        """
        if self.is_int(item):
            res_set.add(item)
        elif self.is_set(item):
            set_indexes.append(index)
        else:
            empty_indexes.append(index)

    def res_filter(self, nda: np.ndarray) -> (set, list, list):
        # 「当前行/列/宫所有确定的数的集合」
        res_set = set()
        # 未确定的单元格的索引
        set_indexes = []
        # 空单元格的索引
        empty_indexes = []

        # 行/列
        if len(nda.shape) == 1:
            for i, item in enumerate(nda):
                self._filter(item, i, res_set, set_indexes, empty_indexes)
        # 宫
        else:
            for i, row in enumerate(nda):
                for j, item in enumerate(row):
                    self._filter(item, (i, j), res_set, set_indexes, empty_indexes)

        return res_set, set_indexes, empty_indexes

    def resolve(self, row):
        """解数独"""
        row_set, set_indexes, empty_indexes = self.res_filter(row)
        # 如果确定数的集合的长度和未确定单元格数目相加不为9，则无解
        if len(row_set) + len(set_indexes) + len(empty_indexes) != 9:
            raise ValueError('无解')
        # 如果确定的数和「1-9」的集合差集不为空，则无解
        if len(row_set - self.standard_set):
            raise ValueError('无解')
        # 如果当前行/列/宫有值为空，初始化空值
        if empty_indexes:
            for i in empty_indexes:
                # 计算差集，获取当前单元格的值
                res = self.standard_set - row_set
                # 如果结果集只有一个值，说明该行/列/宫已经全部计算出来
                if len(res) == 1:
                    row[i] = res.pop()
                    return
                row[i] = res
                set_indexes.append(i)

        # 处理未确定的单元格
        if set_indexes:
            for i in set_indexes.copy():
                # 计算和确定的单元格之间的差集，获取当前单元格的值
                res = row[i] - row_set
                if not res:
                    raise ValueError("无解")

                if len(res) == 1:
                    row[i] = res.pop()
                    row_set.add(row[i])
                    set_indexes.remove(i)
                    continue

                # 「所有其他格的集合的并集」
                union_set = reduce(ior, (row[_] for _ in set_indexes if _ != i), set())
                # 当前单元格减去「所有其他格的集合的并集」的差集
                u_res = res - union_set
                # 如果长度为1的,可以确定当前的值为差集所得数
                if len(u_res) == 1:
                    row[i] = u_res.pop()
                    row_set.add(row[i])
                    set_indexes.remove(i)
                    continue

                row[i] = res

    def row_solve(self):
        """
        将每一行的结果确定
           a. 当前格为空白处的，值等于「1-9」的集合减去「当前行所有确定的数的集合」
           b. 如果当前格为集合的，且减去「所有其他格的集合的并集」为1的，
              可以确定当前处的值为差集所得数
        :return:
        """
        for row in self.table:
            self.resolve(row)

    def col_solve(self):
        for row in self.table.T:
            self.resolve(row)

    def grid_solve(self):
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                self.resolve(self.table[i:i + 3, j:j + 3])

    def validate(self):
        for i in self.table:
            if not all(map(self.is_int, i)):
                return False
        return True

    def try_num(self):
        origin_table = copy.deepcopy(self.table)
        for r, row in enumerate(origin_table):
            for c, col in enumerate(row):
                if not self.is_set(col):
                    continue
                for i in col:
                    self.table = copy.deepcopy(origin_table)
                    self.table[r, c] = i
                    try:
                        self.sudoku_solve()
                    except ValueError:
                        continue
                else:
                    return

    def sudoku_solve(self):
        # 循环2-3遍较为合理
        for i in range(3):
            self.row_solve()
            self.col_solve()
            self.grid_solve()
        # 如果校验未通过，则数独需要通过试数解决
        if not self.validate():
            self.try_num()
        else:
            res = self.table.tolist()
            self.results.append(res)
            if self.ONLY_ONE:
                raise StopIteration('停止求解')

    def run(self):
        try:
            self.sudoku_solve()
        except ValueError:
            pass
        except StopIteration:
            pass
        print(f"该数独有「{len(self.results)}」个解")
        print(',\n'.join((str(i) for i in self.results)), end=f'\n{"-" * 30}\n')
        return self.results

    def query_all(self):
        # 求所有解
        return self.run()

    def query_one(self):
        # 只求一个解
        self.ONLY_ONE = True
        return self.run()


def test():
    s = SudokuHandler()
    a = [
        [1, None, 3, None, None, None, 9, None, None],
        [None, None, 7, None, 4, None, 5, 2, None],
        [8, None, 4, None, 5, None, None, None, None],
        [None, 6, None, None, 7, None, None, 5, None],
        [None, None, None, None, 1, None, None, None, 3],
        [5, None, None, 2, 8, None, None, None, None],
        [None, None, None, None, 2, None, 3, None, 6],
        [7, None, None, 1, None, None, None, 8, None],
        [None, 4, None, None, 3, None, None, 1, None]
    ]


    s.build_from_list(a)
    # s.query_all()
    s.query_one()
    # print(s.table.tolist())

if __name__ == '__main__':
    test()

