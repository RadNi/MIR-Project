class Printer:
    tab_depth = 0
    tab_str = ""

    @staticmethod
    def indent_right():
        Printer.tab_depth += 1
        Printer.tab_str += "\t"

    @staticmethod
    def indent_left():
        if Printer.tab_depth > 0:
            Printer.tab_depth -= 1
            Printer.tab_str = Printer.tab_str[:-1]

    @staticmethod
    def print(msg):
        print(Printer.tab_str + msg)
