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
    def print(msg: str):
        msg_lines = msg.split('\n')
        for line in msg_lines:
            print(Printer.tab_str + line.lstrip())
