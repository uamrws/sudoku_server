import collections
import copy
from functools import reduce

import numpy as np

from sudoku_server.utils.common import grid_property, get_grid_by_xy, get_xy_by_grid


class SudokuHandler(object):

    def __init__(self):
        self.base_set = set(range(1, 10))
        self.table = None
        # 已探明
        self.is_ascertain = lambda x: (isinstance(x, int) and 0 < x < 10)
        # 未探明
        self.is_uncertain = lambda x: isinstance(x, set)
        # 未初始
        self.is_uninit = lambda x: x is None
        # 待探明「为探明中长度为1」
        self.wait_ascertain = lambda x: len(x) == 1

        # 缓存每行/列/宫所有已探明和未探明的单元格的hash_map
        # 键分为三个部分组成，举例，第一行的探明键为"asc_row1"
        self.cache_hash_map = {

        }
        # 所有结果集
        self.results = []
        self.ONLY_ONE = False

    def build(self, a_list: list):
        """从表格导入数据"""
        self.table = np.array(a_list, dtype=object)
        return self

    @property
    def rows(self):
        return self.table

    @property
    def cols(self):
        return self.table.T

    @grid_property
    def grids(self):
        """
        可以根据「行,列」下标 (x,y) 来获取对应宫
        也可以根据下标 x 获取第 x 宫
        或者根据下标 [x:y] 获取第 x 宫到第 y 宫的列表
        """
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                yield self.table[i:i + 3, j:j + 3]

    @staticmethod
    def get_row_key(row, _type):
        return f"{_type}_row{row}"

    @staticmethod
    def get_col_key(col, _type):
        return f"{_type}_col{col}"

    @staticmethod
    def get_grid_key(grid, _type):
        return f"{_type}_grid{grid}"

    def get_cache_keys(self, row, col, _type):
        grid = get_grid_by_xy(row, col)
        if _type == 'both':
            keys = (
                self.get_row_key(row, "unc"),
                self.get_col_key(col, "unc"),
                self.get_grid_key(grid, "unc"),
                self.get_row_key(row, "asc"),
                self.get_col_key(col, "asc"),
                self.get_grid_key(grid, "asc")
            )
        else:
            keys = (
                self.get_row_key(row, _type),
                self.get_col_key(col, _type),
                self.get_grid_key(grid, _type),
            )
        return keys

    def op_cache(self, method, key, item=None):
        if method == "POST":
            self.cache_hash_map[key] = item
        elif method == "GET":
            return self.cache_hash_map.get(key, None)
        elif method == "DELETE":
            self.cache_hash_map.pop(key, None)
        return True

    def storage_cache(self, key, item):
        """缓存行列宫所有 已探明/未探明的单元格"""
        return self.op_cache("POST", key, item)

    def delete_all_cache(self, row, col, _type):
        """删除行列宫的缓存"""
        keys = self.get_cache_keys(row, col, _type)

        for key in keys:
            self.op_cache("DELETE", key)
        return True

    def get_cache(self, key):
        """获取行列宫的缓存"""
        return self.op_cache("GET", key)

    @staticmethod
    def _get_abs_grid_index(grid, idx):
        """根据「宫数」推断「一维相对下标」的「二维绝对坐标」"""
        # 计算一维中的「宫的起始单元格」的二维位置
        base_row, base_col = get_xy_by_grid(grid)
        # 计算idx与基准的的相对位置relative
        rla_row = idx // 3
        rla_col = idx % 3
        # 返回idx在整体中的绝对位置
        return base_row + rla_row, base_col + rla_col

    def get_idxes_by_key(self, items, key):
        """
        根据key返回
            1.其他未解答出的单元格在当前「行/列/宫」的「相对索引」
            2.已解答出的单元格在当前「行/列/宫」的「相对索引」
        """
        idxes = self.get_cache(key)
        if idxes is None:
            map_func = self.is_ascertain if key[:3] == 'asc' else self.is_uncertain
            bool_idx = np.array(list(map(map_func, items)))
            # 获取当前的下标值
            idxes = np.argwhere(bool_idx == 1)
            idxes = idxes.reshape(len(idxes)).astype(object)
            self.storage_cache(key, idxes)
        return list(idxes)

    def intersection_removel(self):
        """区块删减法(Intersection Removel)"""
        pass

    def x_wing(self, row, col):
        """矩形对角线删减法(X-wing)"""
        pass

    def swordfish(self):
        """三链数删减法(Swordfish)"""
        pass

    def xy_wing(self):
        """XY 形态匹配法(XY-wing)"""
        pass

    def try_num(self):
        """试探法"""
        pass

    def _filter_equal_idxes(self, unc_idxes, cur_item):
        equal_idxes_map = collections.defaultdict(set)
        for i, idx in enumerate(unc_idxes):
            item = cur_item[idx]
            k = frozenset(item)
            # 显示唯一
            if self.wait_ascertain(item):
                equal_idxes_map[k].add(idx)
                continue
            for j in unc_idxes[i:]:
                if item != cur_item[j]:
                    continue
                equal_idxes_map[k].update((idx, j))
            else:
                # 如果候选数小于单元格数目，则数独错误无解
                if len(k) < len(equal_idxes_map[k]):
                    raise ValueError("此题无解")
                # 如果候选数大于单元格数目，则不符合显示规则
                elif len(k) > len(equal_idxes_map[k]):
                    equal_idxes_map.pop(k)
        return equal_idxes_map

    def gen_abc_naked(self, key, cur_item):
        """显式抽象"""
        # 处理其他显式
        # 规则：某行/列/宫中的某「n」个单元格格的候选数恰好为「n」个
        unc_idxes = self.get_idxes_by_key(cur_item, key)
        if not len(unc_idxes):
            return

        equal_idxes_map = self._filter_equal_idxes(unc_idxes, cur_item)
        if not equal_idxes_map:
            return False

        for set_k, set_idxes in equal_idxes_map.items():
            # 如果是所有单元格都相等，则没必要进行下去了
            if len(set_idxes) == len(unc_idxes):
                return True

            # 如果「不是所有的单元格都相等」
            # 则按规则从「其他单元格」删除「候选数tup_k」
            for i in unc_idxes:
                if i in set_idxes:
                    continue
                item = cur_item[i]
                if not item & set_k:
                    continue
                item = cur_item[i]
                item -= set_k
                if not item:
                    raise ValueError("此题无解")
                yield i

    def row_naked(self, row, flag=0):
        """行显式"""
        cur_row = self.rows[row]

        key = self.get_row_key(row, 'unc')

        l_change_idxes = set(self.gen_abc_naked(key, cur_row))
        if l_change_idxes:
            l_change_idxes.update(self.row_naked(row, flag + 1))
        if flag:
            return l_change_idxes

        for col, grid in [(i, get_grid_by_xy(row, i)) for i in l_change_idxes if i < row]:
            self.col_naked(col)
            if grid >= row:
                continue
            self.grid_naked(grid)

    def col_naked(self, col, flag=0):
        """列显式"""
        cur_col = self.cols[col]

        key = self.get_col_key(col, 'unc')

        l_change_idxes = set(self.gen_abc_naked(key, cur_col))
        if l_change_idxes:
            l_change_idxes.update(self.col_naked(col, flag + 1))
        if flag:
            return l_change_idxes
        for row, grid in [(i, get_grid_by_xy(i, col)) for i in l_change_idxes if i <= col]:
            self.row_naked(row)
            if grid >= col:
                continue
            self.grid_naked(grid)

    def grid_naked(self, grid, flag=0):
        """宫显式"""
        cur_grid = self.grids[grid].reshape(9)

        key = self.get_grid_key(grid, 'unc')

        l_change_idxes = set(self.gen_abc_naked(key, cur_grid))
        if l_change_idxes:
            l_change_idxes.update(self.grid_naked(grid, flag + 1))
        if flag:
            return l_change_idxes
        for row, col in [self._get_abs_grid_index(grid, i) for i in l_change_idxes]:
            if row <= grid:
                self.row_naked(row)
            if col <= grid:
                self.col_naked(col)

    def _naked(self, num):
        """显式入口"""
        self.row_naked(num)
        self.col_naked(num)
        self.grid_naked(num)

    @staticmethod
    def _filter_inter_idxes(unc_idxes, cur_item):
        """从所有集合的排列组合中选出符合隐式规则的集合以及下标"""
        # 与当前行/列/宫所有符合规则的「交集：下标」map
        inter_idxes_map = collections.defaultdict(set)

        def _filter_comb(_unc_idxes, _cur_item, i=0):
            """
            通过筛选所有的排列组合，
            选出符合隐式规则的集合以及下标
            并把值放到「inter_idxes_map」中
            """
            result = collections.defaultdict(set)

            s_first_idx = _unc_idxes[i]
            s_first_item = _cur_item[s_first_idx]

            result[frozenset(s_first_item)].add(i)
            if len(_unc_idxes[i:]) == 1:
                return result

            result.update(_filter_comb(_unc_idxes, _cur_item, i + 1))
            for tup_k in result.copy():
                set_k = set(tup_k)
                inter = s_first_item & set_k
                if inter:
                    idxes = result[tup_k].copy()
                    idxes.add(i)
                    result[frozenset(inter)].update(idxes)
                    if len(inter) >= len(idxes):
                        diff = inter - set.union(set(),
                                                 *_cur_item[[i for i in _unc_idxes if i not in idxes]])
                        if len(diff) == len(idxes):
                            inter_idxes_map[frozenset(diff)] = idxes
                        if len(diff) > len(idxes):
                            raise ValueError("此题无解")
            return result

        _filter_comb(unc_idxes, cur_item)
        return inter_idxes_map

    def gen_abc_hidden(self, key, cur_item):
        """隐式抽象"""
        unc_idxes = self.get_idxes_by_key(cur_item, key)
        if not len(unc_idxes):
            return
        # 处理隐式规则
        inter_idxes_map = self._filter_inter_idxes(unc_idxes, cur_item)
        # 隐式数集删减法

        # 其他隐式删减法
        # 如果一个键「单元格交集」与其他键「其他单元格交集」的差集等于对应的下标数,
        # 则符合隐式规则，进一步判断

        for set_k, set_idxes in inter_idxes_map.items():
            cp_dict = inter_idxes_map.copy()
            cp_dict.pop(set_k)

            diff_set = set_k - set.union(set(), *cp_dict.keys())

            if len(set_idxes) < len(diff_set):
                # 和显式不同，tup_inter是唯一交集，意味着除当前这些坐标以外，其他单元格都没有这些数
                # 如果这些「单元格的候选数」大于「单元格的数目」，很明显数独无解
                raise ValueError("此题无解")
            elif len(set_idxes) == len(diff_set):
                # 将下标列表中的单元格都去除其他元素
                for i in set_idxes:
                    item = cur_item[i]
                    if item == diff_set:
                        continue
                    item &= diff_set
                    yield i

    def row_hidden(self, row, flag=0):
        """行隐式"""
        cur_row = self.rows[row]

        key = self.get_row_key(row, 'unc')

        l_change_idxes = set(self.gen_abc_hidden(key, cur_row))
        if l_change_idxes:
            l_change_idxes.update(self.row_hidden(row, flag + 1))
        if flag:
            return l_change_idxes
        for col, grid in [(i, get_grid_by_xy(row, i)) for i in l_change_idxes if i < row]:
            self.col_hidden(col)
            if grid >= row:
                continue
            self.grid_hidden(grid)

    def col_hidden(self, col, flag=0):
        """列隐式"""
        cur_col = self.cols[col]

        key = self.get_col_key(col, 'unc')

        l_change_idxes = set(self.gen_abc_hidden(key, cur_col))
        if l_change_idxes:
            l_change_idxes.update(self.col_hidden(col, flag + 1))
        if flag:
            return l_change_idxes
        for row, grid in [(i, get_grid_by_xy(i, col)) for i in l_change_idxes if i <= col]:
            self.row_hidden(row)
            if grid >= col:
                continue
            self.grid_hidden(grid)

    def grid_hidden(self, grid, flag=0):
        """宫隐式"""
        cur_grid = self.grids[grid].reshape(9)

        key = self.get_grid_key(grid, 'unc')

        l_change_idxes = set(self.gen_abc_hidden(key, cur_grid))
        if l_change_idxes:
            l_change_idxes.update(self.grid_hidden(grid, flag + 1))
        if flag:
            return l_change_idxes
        for row, col in [self._get_abs_grid_index(grid, i) for i in l_change_idxes]:
            if row <= grid:
                self.row_hidden(row)
            if col <= grid:
                self.col_hidden(col)

    def _hidden(self, num):
        """隐式入口"""
        self.row_hidden(num)
        self.col_hidden(num)
        self.grid_hidden(num)

    def naked_hidden(self):
        """
        显/隐式入口
        显式隐式互斥
        """
        for num in range(9):
            self._naked(num)
            self._hidden(num)

    def use_ruler(self):
        self.naked_hidden()

    def gen_abc_init(self, s_asc_union, cur_item):
        """初始化抽象生成器"""
        for i in range(9):
            item = cur_item[i]
            if self.is_ascertain(item):
                continue
            if self.is_uncertain(item):
                item -= s_asc_union
            elif self.is_uninit(item):
                item = cur_item[i] = self.base_set - s_asc_union
                if not item:
                    raise ValueError("错误的数独")
                yield i
            else:
                raise ValueError("错误的数独")

    def row_init(self, row):
        """初始化行"""
        cur_row = self.rows[row]
        key = self.get_row_key(row, "asc")

        asc_row_idxes = self.get_idxes_by_key(cur_row, key)
        s_asc_union = set(cur_row[asc_row_idxes])
        for i in self.gen_abc_init(s_asc_union, cur_row):
            self.table[row, i] = cur_row[i]

    def col_init(self, col):
        """初始化列"""
        cur_col = self.cols[col]
        key = self.get_col_key(col, "asc")

        asc_col_idxes = self.get_idxes_by_key(cur_col, key)
        s_asc_union = set(cur_col[asc_col_idxes])
        for i in self.gen_abc_init(s_asc_union, cur_col):
            self.table[i, col] = cur_col[i]

    def grid_init(self, grid):
        """初始化宫"""
        cur_grid = self.grids[grid].reshape(9)
        key = self.get_grid_key(grid, "asc")

        asc_grid_idxes = self.get_idxes_by_key(cur_grid, key)
        s_asc_union = set(cur_grid[asc_grid_idxes])
        for i in self.gen_abc_init(s_asc_union, cur_grid):
            abs_row, abs_col = self._get_abs_grid_index(grid, i)
            self.table[abs_row, abs_col] = cur_grid[i]
            self.delete_all_cache(abs_row, abs_col, 'both')

    def sudoku_init(self):
        """初始化数独"""
        for i in range(9):
            self.row_init(i)
            self.col_init(i)
            self.grid_init(i)

    def resolve(self):
        """
            解答数独
        """
        # 第一步初始化数独，即将每个格子首先根据已经探明的数进行初始化
        # 返回是否有需要进行显示唯一规则的
        self.sudoku_init()

        self.use_ruler()
        return self.table

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

    s.build(a)
    s.resolve()

if __name__ == '__main__':
    test()
