import collections
from functools import reduce
from typing import List

import numpy as np

from sudoku_server.utils.common import grid_property, get_grid_by_xy, get_xy_by_grid


class CacheDict(dict):
    def __missing__(self, key):
        self[key] = value = CacheDict()
        return value


class sudoku_cell(object):
    def __init__(self, sudoku_obj, row, col):
        self._base = {1, 2, 3, 4, 5, 6, 7, 8, 9}
        self.row = row
        self.col = col
        self.sudoku_obj = sudoku_obj
        self.grid = grid = get_grid_by_xy(row, col)
        with sudoku_obj.cache_factory(row, col, grid) as cache_obj:
            self.row_asc = cache_obj.row_asc
            self.col_asc = cache_obj.col_asc
            self.grid_asc = cache_obj.grid_asc
        self.frozen()

    def frozen(self):
        """冻结当前的状态，用以后续判断是否有变化"""
        self.__base = self._base - (self.row_asc | self.col_asc | self.grid_asc)
        self.__value = 1 << 9 | reduce(lambda x, y: x | (1 << (y - 1)), self.__base, 0)
        self.row_asc_len = len(self.row_asc)
        self.col_asc_len = len(self.col_asc)
        self.grid_asc_len = len(self.grid_asc)
        self._base_len = len(self._base)

    def __len__(self):
        return len(self.base)

    def __repr__(self):
        return repr(self.base)
        # return f"{repr(self.base)}: {self.value}"

    def __iter__(self):
        return iter(self.base)

    def __int__(self):
        return np.int(self.value)

    def __bool__(self):
        return bool(self.base)

    def __lt__(self, other):
        return self.value < other

    def __gt__(self, other):
        return self.value > other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __hash__(self):
        return hash(self.value)

    def __or__(self, other):
        return self.base | other

    def __and__(self, other):
        return self.base & other

    def __xor__(self, other):
        return self.base ^ other

    def __ror__(self, other):
        return self.base | other

    def __rand__(self, other):
        return self.base & other

    def __rxor__(self, other):
        return self.base ^ other

    def __sub__(self, other):
        return self.base - other

    def __rsub__(self, other):
        return other - self.base

    def __isub__(self, other):
        self._base -= other
        return self

    def __ixor__(self, other):
        self._base ^= other
        return self

    def __ior__(self, other):
        self._base |= other
        return self

    def __iand__(self, other):
        self._base &= other
        return self

    def __contains__(self, item):
        return item in self.base

    @property
    def unchanged(self):
        """判断是否有变化"""
        return (
                self.row_asc_len == len(self.row_asc)
                and self.col_asc_len == len(self.col_asc)
                and self.grid_asc_len == len(self.grid_asc)
                and self._base_len == len(self._base)
        )

    @property
    def base(self) -> set:
        if self.unchanged:
            return self.__base
        self.frozen()
        return self.__base

    @property
    def value(self) -> int:
        if self.unchanged:
            return self.__value
        self.frozen()
        return self.__value

    @property
    def is_ascertain(self):
        return len(self.base) == 1

    def refresh(self):
        """将当前唯一解的单元格更新"""
        assert self.is_ascertain
        value = self.base.pop()
        with self.sudoku_obj.cache_factory(
                self.row, self.col, self.grid
        ) as cache_obj:
            if self.sudoku_obj.is_initialized:
                cache_obj.row_unc.remove(self)
                cache_obj.col_unc.remove(self)
                cache_obj.grid_unc.remove(self)
            cache_obj.row_asc.add(value)
            cache_obj.col_asc.add(value)
            cache_obj.grid_asc.add(value)
        self.sudoku_obj.table[self.row, self.col] = value
        return value

    def remove(self, item):
        self._base.remove(item)


class AscCache(set):
    def pop(self):
        raise PermissionError("can not pop item'")


class UncCache(object):
    def __init__(self, nums):
        assert isinstance(nums, np.ndarray)
        self.nums = nums
        self._nums: list = nums.tolist()

    def __getitem__(self, item):
        return self.nums[item]

    def __setitem__(self, key, value):
        self.nums[key] = value

    def __and__(self, other):
        return self.nums & other

    def __eq__(self, other):
        return self.nums == other

    def __lt__(self, other):
        return self.nums < other

    def __gt__(self, other):
        return self.nums > other

    def __ne__(self, other):
        return self.nums != other

    def __repr__(self):
        return repr(self.nums)

    def remove(self, item):
        self._nums.remove(item)
        self.nums = np.array(self._nums)


class SudokuCacheFactory:
    """缓存工厂，用以缓存每行/列/宫的「已探明」的值"""
    __ATTRS = (
        "row_asc",
        "col_asc",
        "grid_asc",
        "row_unc",
        "col_unc",
        "grid_unc",
    )

    def __init__(self, handler):
        self.handler = handler
        self.__ObjMap = CacheDict()

    def __call__(self, row=None, col=None, grid=None):
        self.row = row
        self.col = col
        self.grid = grid
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.row = self.col = self.grid = None

    def __getattr__(self, item):
        if item not in self.__ATTRS:
            raise AttributeError(f"SudokuCache Object Has No Attribute: {item}")
        attr, _type = item.split("_")

        self_attr = getattr(self, attr)

        if self_attr is None:
            raise NotImplementedError

        cache_dict = self.__ObjMap[attr][_type]
        handler_attrs = getattr(self.handler, f"{attr}s")

        if self_attr in cache_dict:
            return cache_dict[self_attr]

        _ = handler_attrs[self_attr]
        if _type == "asc":
            nums = _[(_ <= 9) & (_ > 0)]
            obj = AscCache(nums)
        else:
            nums = _[_ > 9]
            obj = UncCache(nums)
        cache_dict[self_attr] = obj
        return obj


class SudokuHandler:

    def __init__(self):
        self.table = None
        self.is_initialized = False
        self.cache_factory = SudokuCacheFactory(self)
        # 所有结果集
        self.results = []
        self.ONLY_ONE = False

    @classmethod
    def build(cls, a_list: list):
        """从表格导入数据"""
        obj = cls()
        obj.table = np.array(a_list, dtype=object)
        return obj

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
    def _get_abs_grid_index(grid, idx):
        """根据「宫数」推断「一维相对下标」的「二维绝对坐标」"""
        # 计算一维中的「宫的起始单元格」的二维位置
        base_row, base_col = get_xy_by_grid(grid)
        # 计算idx与基准的的相对位置relative
        rla_row = idx // 3
        rla_col = idx % 3
        # 返回idx在整体中的绝对位置
        return base_row + rla_row, base_col + rla_col

    def intersection_remove(self):
        """区块删减法(Intersection Remove)"""
        flag = False
        unc_cells = self.table[self.table > 9]
        if not unc_cells.size:
            return flag
        _map = collections.defaultdict(set)
        for cell in unc_cells:
            assert isinstance(cell, sudoku_cell)
            if not cell:
                raise ValueError("此数独无解")
            for i in cell:
                # 某一个宫中出现过数值i的行下标
                _map[f"{cell.grid}-grid-{i}-row"].add(cell.row)
                # 某一个宫中出现过数值i的列下标
                _map[f"{cell.grid}-grid-{i}-col"].add(cell.col)
                # 某一个行中出现过数值i的宫下标
                _map[f"{cell.row}-row-{i}-grid"].add(cell.grid)
                # 某一个列中出现过数值i的宫下标
                _map[f"{cell.col}-col-{i}-grid"].add(cell.grid)
        with self.cache_factory() as cache_obj:
            for k, v in _map.items():
                if len(v) > 1:
                    continue
                serial, _type, num, attr = k.split('-')
                num = int(num)
                serial = int(serial)
                setattr(cache_obj, attr, *v)
                unc = getattr(cache_obj, f"{attr}_unc")
                for i in unc:
                    if getattr(i, _type) == serial:
                        continue
                    if num in i:
                        i.remove(num)
                        flag = True
        return flag

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

    def _naked(self, cell, unc):
        equal = unc[unc == cell]
        assert equal.size
        if equal.size == 1:
            return False
        if equal.size > len(cell):
            raise ValueError("此数独无解")
        if equal.size == len(cell):
            flag = False
            for i in unc:
                if i == cell:
                    continue
                if i & cell:
                    i -= cell
                    flag = True
            return flag

    def naked(self):
        """显式入口"""
        flag = False
        unc_cells = self.table[self.table > 9]
        if not unc_cells.size:
            return flag
        for cell in unc_cells:
            assert isinstance(cell, sudoku_cell)
            if not cell:
                raise ValueError("此数独无解")
            if cell.is_ascertain:
                cell.refresh()
                flag = True
                continue
            with self.cache_factory(cell.row, cell.col, cell.grid) as cache_obj:
                row_unc = cache_obj.row_unc
                col_unc = cache_obj.col_unc
                grid_unc = cache_obj.grid_unc
            if any([
                self._naked(cell, row_unc),
                self._naked(cell, col_unc),
                self._naked(cell, grid_unc),
            ]):
                flag = True
        return flag

    def _hidden(self, unc):
        flag = False
        _ = collections.defaultdict(list)
        for idx, item in enumerate(unc):
            for i in item:
                _[i].append(idx)

        indexes_map = collections.defaultdict(set)
        for k, v in _.items():
            indexes_map[tuple(v)].add(k)

        for k, v in indexes_map.items():
            if len(k) != len(v):
                continue
            for i in k:
                if unc[i].base == v:
                    continue
                unc[i] &= v
                flag = True

        return flag

    def hidden(self):
        """隐式入口"""
        flag = False
        for i in range(9):
            with self.cache_factory(i, i, i) as cache_obj:
                row_unc = cache_obj.row_unc
                col_unc = cache_obj.col_unc
                grid_unc = cache_obj.grid_unc
            if any([
                self._hidden(row_unc),
                self._hidden(col_unc),
                self._hidden(grid_unc),
            ]):
                flag = True

        return flag

    def use_ruler(self):
        while any([
            self.hidden(),
            self.intersection_remove(),
            self.naked(),
        ]):
            pass

    def init_cell(self, row, col):
        cur_cell = sudoku_cell(self, row, col)
        if not cur_cell:
            raise ValueError("错误的数独")
        elif cur_cell.is_ascertain:
            cur_cell.refresh()
        else:
            self.table[row, col] = cur_cell

    def sudoku_init(self):
        """初始化数独"""
        try:
            if np.any((self.table > 9) | (self.table < 0)):
                raise ValueError("错误的数独")
        except TypeError:
            raise ValueError("错误的数独")
        for row, col in np.argwhere(self.table == 0).astype(object):
            self.init_cell(row, col)
        self.is_initialized = True

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
    a = [
        [1, 0, 3, 0, 0, 0, 9, 0, 0],
        [0, 0, 7, 0, 4, 0, 5, 2, 0],
        [8, 0, 4, 0, 5, 0, 0, 0, 0],
        [0, 6, 0, 0, 7, 0, 0, 5, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 3],
        [5, 0, 0, 2, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 3, 0, 6],
        [7, 0, 0, 1, 0, 0, 0, 8, 0],
        [0, 4, 0, 0, 3, 0, 0, 1, 0]
    ]
    s = SudokuHandler.build(a)
    s.resolve()
    # print(s.resolve().tolist())


if __name__ == '__main__':
    test()
    # print(s.table.tolist() == [[1, 5, 3, 8, 6, 2, 9, 7, 4], [6, 9, 7, 3, 4, 1, 5, 2, 8], [8, 2, 4, 9, 5, 7, 6, 3, 1],
    #                            [3, 6, 8, 4, 7, 9, 1, 5, 2],
    #                            [4, 7, 2, 5, 1, 6, 8, 9, 3], [5, 1, 9, 2, 8, 3, 4, 6, 7], [9, 8, 1, 7, 2, 5, 3, 4, 6],
    #                            [7, 3, 6, 1, 9, 4, 2, 8, 5],
    #                            [2, 4, 5, 6, 3, 8, 7, 1, 9]])
    # print([[1, 5, 3, 8, 6, 2, 9, 7, 4], [6, 9, 7, 3, 4, 1, 5, 2, 8], [8, 2, 4, 9, 5, 7, 6, 3, 1],
    #        [3, 6, 8, 4, 7, 9, 1, 5, 2],
    #        [4, 7, 2, 5, 1, 6, 8, 9, 3], [5, 1, 9, 2, 8, 3, 4, 6, 7], [9, 8, 1, 7, 2, 5, 3, 4, 6],
    #        [7, 3, 6, 1, 9, 4, 2, 8, 5],
    #        [2, 4, 5, 6, 3, 8, 7, 1, 9]] == [[1, 5, 3, 8, 6, 2, 9, 7, 4], [6, 9, 7, 3, 4, 1, 5, 2, 8],
    #                                         [8, 2, 4, 9, 5, 7, 6, 3, 1], [3, 6, 8, 4, 7, 9, 1, 5, 2],
    #                                         [4, 7, 2, 5, 1, 6, 8, 9, 3], [5, 1, 9, 2, 8, 3, 4, 6, 7],
    #                                         [9, 8, 1, 7, 2, 5, 3, 4, 6], [7, 3, 6, 1, 9, 4, 2, 8, 5],
    #                                         [2, 4, 5, 6, 3, 8, 7, 1, 9]]
    #       )
    #
