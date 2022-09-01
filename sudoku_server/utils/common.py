"""
规则 1
    显式唯一候选数法(Naked Single)：若某行，或列，
或九宫格中的某个空格候选数唯一，那么该候选数就填该
空格。

规则 2
    隐式唯一候选数法(Hidden Single)：若某个候选
数只出现在某行，或列，或九宫格的某个空格中，那么该空
格就填该候选数。

规则 3
    显式二数集删减法(Naked Pair)：若某行，或列，
或九宫格中的某 2 个空格的候选数恰好为 2 个，那么这 2 个
候选数可以从该行，或列，或九宫格的其他空格候选数中
删除。

规则 4
    隐式二数集删减法(Hidden Pair)：若某 2 个候选
数只出现在某行，或列，或九宫格中的某 2 个空格中，那么
这 2 个空格中不同于这 2 个候选数的其他候选数可删除。

规则 5
    显式三数集删减法(Naked Triplet)：若某行，或列，
或九宫格中的某 3 个空格的候选数恰好为 3 个，那么这 3 个
候选数可以从该行，或列，或九宫格的其他空格候选数中 删除。

规则 6
    隐式三数集删减法(Hidden Triplet)：若某 3 个候
选数只出现在某行，或列，或九宫格中的某 3 个空格中，那
么这 3 个空格中不同于这 3 个候选数的其他候选数可删除。

规则 7
    显式四数集删减法(Naked Quad)：若某行，或列，
或九宫格中的某 4 个空格的候选数恰好为 4 个，那么这 4 个
候选数可以从该行，或列，或九宫格的其他空格候选数中
删除。

规则 8
    隐式四数集删减法(Hidden Quad)：若某 4 个候选
数只出现在某行，或列，或九宫格中的某 4 个空格中，那么
这 4 个空格中不同于这 4 个候选数的其他候选数可删除。

规则 9
    区块删减法(Intersection Removel)
区块对行的影响：在某一区块中，当所有可能出现某个
数字的单元格都位于同一行时，就可以把这个数字从该行的
其他单元格的候选数中删除。
区块对列的影响：在某一区块中，当所有可能出现某个
数字的单元格都位于同一列时，就可以把这个数字从该列的
其他单元格的候选数中删除。
行或列对区块的影响：在某一行(列)中，当所有可能出
现某个数字的单元格都位于同一区块中时，就可以把这个数
字从该区块的其他单元格的候选数中删除。

规则 10
    矩形对角线删减法(X-wing)：如果一个数字正好
出现且只出现在某 2 行(列)相同的 2 列(行)上，则这个数字就
可以从这 2 列(行)上其他单元格的候选数中删除。

规则 11
    三链数删减法(Swordfish)：如果某个数字在某
3 列(行)中只出现在相同的 3 行(列)中，则这个数字将从这
3 行(列)上其他的候选数中删除。

规则 12
    XY 形态匹配法(XY-wing)：若 XY 和 YZ 同在一
个区块但不同行(列)中，而 XZ 和 XY 在同一行(列)，但在不
同区块中。那么在 XY 所在区块中与 XY 所在行(列)交集空
格中应该删除候选数 Z，并且在 XZ 所在区块与 YZ 所在行
(列)交集的空格中应该删除候选数 Z。其中，XY、YZ、XZ
分别是三空格的候选数，并且这三空格没有其他候选数。

规则 13
    XYZ 形态匹配法(XYZ-wing)：若某区块某空格
候选数为 XYZ，在该同区块但不同列(行)的某空格候选数为
YZ，且与 XYZ 所在空格同列(行)但不同区块某空格候选数为
XZ，那么 XYZ 所在区块与 XZ 所在列(行)的交集中的空格候
选数不能为 Z。

规则 14
    试探法：若某个空格的候选数只有 2 个时，进
行试探填写其中一个候选数，若填写成功则该试探成功，若
导致矛盾则另外一个候选数应填该空格。

"""


def get_grid_by_xy(row, col):
    """根据行列计算出宫索引"""
    return row // 3 * 3 + col // 3


def get_xy_by_grid(grid):
    return grid // 3 * 3, grid % 3 * 3


class grid_property(object):
    def __init__(self, func):
        self.func = func
        self.obj = None

    def __get__(self, obj, owner):
        self.obj = obj
        return self

    def __iter__(self):
        """"""
        return self.func(self.obj)

    def __getitem__(self, item):
        """"""
        if isinstance(item, slice):
            return tuple(self.func(self.obj))[item]
        elif isinstance(item, int):
            grid = item
        else:
            if len(item) != 2:
                raise IndexError('tuple length must be 2.')
            row, col = item
            grid = get_grid_by_xy(row, col)
        for item in self.func(self.obj):
            if grid == 0:
                return item
            grid -= 1
