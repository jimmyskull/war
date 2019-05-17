

class ColorFormat:

    def __init__(self, message='', tags=None):
        self.tags = tags if tags else list()
        self.message = message

    def __str__(self):
        tags = ';'.join([str(tag) for tag in self.tags])
        return f'\033[{tags}m{self.message}\033[0m'

    @property
    def bold(self):
        return ColorFormat(self.message, tags=self.tags + [1])

    @property
    def dark_gray(self):
        return ColorFormat(self.message, tags=self.tags + [38, 5, 240])

    @property
    def light_gray(self):
        return ColorFormat(self.message, tags=self.tags + [90])

    @property
    def bottle_green(self):
        return ColorFormat(self.message, tags=self.tags + [96])

    @property
    def green(self):
        return ColorFormat(self.message, tags=self.tags + [32])

    @property
    def cyan(self):
        return ColorFormat(self.message, tags=self.tags + [96])

    @property
    def magenta(self):
        return ColorFormat(self.message, tags=self.tags + [35])

    @property
    def yellow(self):
        return ColorFormat(self.message, tags=self.tags + [33])
