from ply import lex

reserved = {"if": "IF", "defun": "DEFUN", "setq": "SETQ", "t": "T", "nil": "NIL"}
tokens = (
    "LPAREN",
    "RPAREN",
    "IDENTIFIER",
    "NUMBER",
    "STRING",
    "EQUALS",
    "PLUS",
    "MINUS",
    "MULTIPLY",
    "DIVIDE",
    "GREATER",
    "LESS",
    "GREATEREQUAL",
    "LESSEQUAL",
) + tuple(reserved.values())

t_LPAREN = r"\("
t_RPAREN = r"\)"
t_EQUALS = r"="
t_PLUS = r"\+"
t_MINUS = r"-"
t_MULTIPLY = r"\*"
t_DIVIDE = r"/"
t_GREATER = r">"
t_LESS = r"<"
t_GREATEREQUAL = r">="
t_LESSEQUAL = r"<="

t_ignore = " \t\n"


def t_IDENTIFIER(t):
    r"[a-zA-Z][a-zA-Z0-9\-]*"  # identifiers start with a letter and can contain digits and hyphens
    t.type = reserved.get(t.value.lower(), "IDENTIFIER")
    return t


def t_NUMBER(t):
    r"\d*\.?\d+"  # accepting integer as well as floating point numbers
    if "." in t.value:
        t.value = float(t.value)
    else:
        t.value = int(t.value)
    return t


def t_STRING(t):
    r"\"[^\"]*\" "  # accepting quoted strings, and matching any character which is not a double quote
    t.value = t.value[1 : len(t.value) - 1]  # removing the double quotes
    return t


def t_error(t):
    print("Illegal character: ", t.value[0])
    t.lexer.skip(1)


lexer = lex.lex()
