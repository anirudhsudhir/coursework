from ply import lex

tokens = (
    "LPAREN",
    "RPAREN",
    "IF",
    "DEFUN",
    "SETQ",
    "IDENTIFIER",
    "NUMBER",
    "STRING",
    "T",
    "NIL",
)

t_LPAREN = r"\("
t_RPAREN = r"\)"

t_ignore = " \t\n"

reserved = {"if": "IF", "defun": "DEFUN", "setq": "SETQ", "t": "T", "nil": "NIL"}


def t_IDENTIFIER(self, t):
    r"[a-zA-Z][a-zA-Z0-9\-]*"
    t.type = self.reserved.get(t.value.lower(), "IDENTIFIER")
    return t


def t_NUMBER(self, t):
    r"\d*\.?\d+"
    t.value = float(t.value) if "." in t.value else int(t.value)
    return t


def t_STRING(self, t):
    r"\"[^\"]*\" "
    t.value = t.value[1:-1]  # Remove quotes
    return t


def t_error(self, t):
    print(f"Illegal character '{t.value[0]}'")
    t.lexer.skip(1)
