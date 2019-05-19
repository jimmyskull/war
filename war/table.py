"""Terminal table."""
from war.cformat import ColorFormat


UNICODE_BOX_DRAWING = {
    # Separator
    'horz_sep': '─',
    'vert_sep': '│',
    # Up
    'up_left': '╭',
    'up_right': '╮',
    'up_div': '┬',
    # Down
    'down_left': '╰',
    'down_right': '╯',
    'down_div': '┴',
    # Middle
    'middle_left': '├',
    'middle_right': '┤',
    'middle_div': '┼',
}


NO_BOX_DRAWING = {
    # Separator
    'horz_sep': '',
    'vert_sep': '',
    # Up
    'up_left': '',
    'up_right': '',
    'up_div': '',
    # Down
    'down_left': '',
    'down_right': '',
    'down_div': '',
    # Middle
    'middle_left': '',
    'middle_right': '',
    'middle_div': '',
}


class Cell:
    """
    Table cell.

    Parameters
    ----------
    value : object
        An object convertible to string.
    attr: list of str, optional, default: depends on value type
        Default is ['ljust'] for flot, ['rjust'] otherwise.
    """

    def __init__(self, value, attr=None):
        self.value = value
        self._attr = None
        self.attr = attr

    def __str__(self):
        return self.format(length=0)

    def _has_alignment(self):
        return set(['center', 'ljust', 'rjust']).intersection(self.attr)

    @property
    def attr(self):
        """List of attributes."""
        return self._attr

    @attr.setter
    def attr(self, value):
        if value is None:
            value = list()
        elif not isinstance(value, list):
            raise ValueError((f'cell attr must be none or a list of str, '
                              f'got {type(value)}'))
        self._attr = value
        if not self._has_alignment():
            try:
                float(self.value)
            except ValueError:
                self._attr += ['ljust']
            else:
                self._attr += ['rjust']

    def format(self, length=0):
        """
        Return the string formatted to be inserted into the table.

        Return
        ------
        length : int
            The length of the cell.
        """
        if isinstance(self.attr, list):
            fmt = ColorFormat(self.value)
            for attr in self.attr:
                if not hasattr(fmt, attr):
                    if attr not in ['center', 'ljust', 'rjust']:
                        raise ValueError(f'cell format {attr!r} not supported')
                    fmt.text = getattr(fmt.text, attr)(length)
                else:
                    fmt = getattr(fmt, attr)
            return str(fmt)
        return self.value


class Table:
    """Print a terminal table."""

    def __init__(self, table_symbol=None):
        if table_symbol is None:
            table_symbol = UNICODE_BOX_DRAWING
        self.header = list()
        self.rows = list()
        self.col_lens = list()
        self.table_symbol = table_symbol

    def _get_row(self, row):
        buf = list()
        for clen, cell in zip(self.col_lens, row):
            buf.append(' ' + cell.format(length=clen) + ' ')
        sep = self.table_symbol['vert_sep']
        buf = sep + sep.join(buf) + sep
        return ''.join(buf)

    def _get_box(self, left, inner, right):
        row = list()
        for clen in self.col_lens:
            row.append(self.table_symbol['horz_sep'] * (clen + 2))
        row = self.table_symbol[inner].join(row)
        row = self.table_symbol[left] + row + self.table_symbol[right]
        return ''.join(row)

    def _compute_lengths(self):
        lens = [len(col.value) for col in self.header]
        for row in self.rows:
            for j, cell in enumerate(row):
                lens[j] = max(lens[j], len(cell.value))
        self.col_lens = lens

    def set_header(self, columns):
        """Set table header."""
        self.header = [Cell(col, attr=['bold']) for col in columns]

    def add_row(self, values, attr=None):
        """
        Append a row into the table.

        Add attr to each cell's attr of the row.

        Parameters
        ----------
        values : list of str and cell
            Column values for the row.  Can be a mix of string and
            Cell values.
        attr : list of str
            Attributes to be appended to the cells' attributes.
        """
        if not self.header:
            raise ValueError('Set up header first.')
        if len(values) != len(self.header):
            raise ValueError(
                'Received {} values in row for {} columns.'.format(
                    len(values), len(self.header)))
        if not attr:
            attr = list()
        row = list()
        for val in values:
            if isinstance(val, Cell):
                val.attr += attr
                row.append(val)
            else:
                row.append(Cell(val, attr=attr))
        self.rows.append(row)

    def format(self):
        """Return string with the formatted table."""
        self._compute_lengths()
        output = list()
        output.append(self._get_box('up_left', 'up_div', 'up_right'))
        output.append(self._get_row(self.header))
        output.append(
            self._get_box('middle_left', 'middle_div', 'middle_right'))
        for row in self.rows:
            output.append(self._get_row(row))
        output.append(self._get_box('down_left', 'down_div', 'down_right'))
        # Remove empty lines, in case of no box drawing.
        output = [line for line in output if line.strip()]
        return '\n'.join(output)
