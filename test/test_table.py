import pytest

from war.table import TerminalTable, Cell, NO_BOX_DRAWING


def test_empty_table():
    table = TerminalTable()
    table.set_header(['a', 'b', 'c'])
    content = table.format()
    assert content == (
        '╭───┬───┬───╮\n'
        '│ \033[1ma\033[0m │ \033[1mb\033[0m │ \033[1mc\033[0m │\n'
        '├───┼───┼───┤\n'
        '╰───┴───┴───╯'
    )


def test_table():
    table = TerminalTable()
    table.set_header(['a', 'b', 'c'])
    table.add_row(['1', '2', '3'])
    content = table.format()
    assert content == (
        '╭───┬───┬───╮\n'
        '│ \033[1ma\033[0m │ \033[1mb\033[0m │ \033[1mc\033[0m │\n'
        '├───┼───┼───┤\n'
        '│ 1 │ 2 │ 3 │\n'
        '╰───┴───┴───╯'
    )


def test_table_header_width():
    table = TerminalTable()
    table.set_header(['aaaa', 'b', 'ccccc'])
    table.add_row([Cell('1', attr=['rjust']), '2', Cell('3', attr=['rjust'])])
    content = table.format()
    assert content == (
        '╭──────┬───┬───────╮\n'
        '│ \033[1maaaa\033[0m │ \033[1mb\033[0m │ \033[1mccccc\033[0m │\n'
        '├──────┼───┼───────┤\n'
        '│    1 │ 2 │     3 │\n'
        '╰──────┴───┴───────╯'
    )


def test_table_ljust():
    table = TerminalTable()
    table.set_header(['aaaa', 'b', 'c'])
    table.add_row([Cell('a', attr=['ljust']), '222', '333333'])
    content = table.format()
    assert content == (
        '╭──────┬─────┬────────╮\n'
        '│ \033[1maaaa\033[0m │ \033[1mb  \033[0m │ \033[1mc     \033[0m │\n'
        '├──────┼─────┼────────┤\n'
        '│ a    │ 222 │ 333333 │\n'
        '╰──────┴─────┴────────╯'
    )


def test_table_rjust():
    table = TerminalTable()
    table.set_header(['aaaa', 'b', 'c'])
    table.add_row([Cell('a', attr=['rjust']), '222', '333333'])
    content = table.format()
    assert content == (
        '╭──────┬─────┬────────╮\n'
        '│ \033[1maaaa\033[0m │ \033[1mb  \033[0m │ \033[1mc     \033[0m │\n'
        '├──────┼─────┼────────┤\n'
        '│    a │ 222 │ 333333 │\n'
        '╰──────┴─────┴────────╯'
    )


def test_table_no_box_drawing():
    table = TerminalTable(NO_BOX_DRAWING)
    table.set_header(['aaaa', 'b', 'c'])
    table.add_row([Cell('a', attr=['rjust']), '222', '333333'])
    content = table.format()
    assert content == (
        ' \033[1maaaa\033[0m  \033[1mb  \033[0m  \033[1mc     \033[0m \n'
        '    a  222  333333 '
    )


def test_cell():
    cell = Cell('val')
    assert str(cell) == 'val'


def test_cell_attr_error():
    msg = "cell attr must be none or a list of str, got <class 'str'>"
    with pytest.raises(ValueError, match=msg):
        Cell('val', 'bold')


def test_cell_attr():
    cell = Cell('val', ['bold'])
    assert str(cell) == '\033[1mval\033[0m'


def test_cell_no_attr():
    cell = Cell('val')
    assert str(cell) == 'val'
