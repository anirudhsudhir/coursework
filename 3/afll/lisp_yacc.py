from ply import yacc

from lisp_lex import tokens


def p_program(p):
    """program : form
    | program form"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = p[1] + [p[2]]


def p_form(p):
    """form : function_definition
    | variable_definition
    | function_call
    | if_condition
    | literal
    | operation"""
    p[0] = p[1]


def p_operation(p):
    """operation : LPAREN operator arguments RPAREN"""
    p[0] = {"type": "operation", "operator": p[2], "arguments": p[3]}


def p_operator(p):
    """operator : EQUALS
    | PLUS
    | MINUS
    | MULTIPLY
    | DIVIDE
    | GREATER
    | LESS
    | GREATEREQUAL
    | LESSEQUAL"""
    p[0] = p[1]


def p_function_definition(p):
    "function_definition : LPAREN DEFUN IDENTIFIER parameter_list body RPAREN"
    p[0] = {
        "type": "function_definition",
        "name": p[3],
        "parameters": p[4],
        "body": p[5],
    }


def p_parameter_list(p):
    "parameter_list : LPAREN parameters RPAREN"
    p[0] = p[2]


def p_parameters(p):
    """parameters :
    | IDENTIFIER parameters"""
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]


def p_body(p):
    """body : form
    | form body"""
    if len(p) == 2:
        p[0] = [p[1]]
    else:
        p[0] = [p[1]] + p[2]


def p_variable_definition(p):
    "variable_definition : LPAREN SETQ IDENTIFIER form RPAREN"
    p[0] = {"type": "variable_definition", "name": p[3], "value": p[4]}


def p_function_call(p):
    "function_call : LPAREN IDENTIFIER arguments RPAREN"
    p[0] = {"type": "function_call", "name": p[2], "arguments": p[3]}


def p_arguments(p):
    """arguments :
    | form arguments"""
    if len(p) == 1:
        p[0] = []
    else:
        p[0] = [p[1]] + p[2]


def p_if_condition(p):
    """if_condition : LPAREN IF test_form then_form RPAREN
    | LPAREN IF test_form then_form else_form RPAREN"""
    if len(p) == 6:
        p[0] = {"type": "if_condition", "test": p[3], "then": p[4], "else": None}
    else:
        p[0] = {"type": "if_condition", "test": p[3], "then": p[4], "else": p[5]}


def p_test_form(p):
    "test_form : form"
    p[0] = p[1]


def p_then_form(p):
    "then_form : form"
    p[0] = p[1]


def p_else_form(p):
    "else_form : form"
    p[0] = p[1]


def p_literal(p):
    """literal : NUMBER
    | STRING
    | T
    | NIL
    | IDENTIFIER"""
    if isinstance(p[1], (int, float)):
        p[0] = {"type": "number", "value": p[1]}
    elif p[1] == "t":
        p[0] = {"type": "boolean", "value": True}
    elif p[1] == "nil":
        p[0] = {"type": "boolean", "value": False}
    elif isinstance(p[1], str):
        if p.slice[1].type == "STRING":
            p[0] = {"type": "string", "value": p[1]}
        else:
            p[0] = {"type": "identifier", "value": p[1]}


def p_error(p):
    if p:
        print("Syntax error at: ", p.value)
    else:
        print("Syntax error at EOF")


parser = yacc.yacc()
while True:
    try:
        s = input("CL > ")
    except EOFError:
        break
    if not s:
        continue
    result = parser.parse(s)
    print(result)
