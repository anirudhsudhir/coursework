```environment = {}

def evaluate(expr, env):
    if isinstance(expr, list):
        return [evaluate(e, env) for e in expr]
    elif isinstance(expr, dict):
        if expr['type'] == 'number':
            return expr['value']
        elif expr['type'] == 'string':
            return expr['value']
        elif expr['type'] == 'boolean':
            return expr['value']
        elif expr['type'] == 'identifier':
            return env.get(expr['value'], f"Undefined variable: {expr['value']}")
        elif expr['type'] == 'variable_definition':
            env[expr['name']] = evaluate(expr['value'], env)
            return env[expr['name']]
        elif expr['type'] == 'function_definition':
            env[expr['name']] = expr
            return f"Function {expr['name']} defined"
        elif expr['type'] == 'function_call':
            func = env.get(expr['name'])
            if not func:
                return f"Undefined function: {expr['name']}"
            params = func['parameters']
            args = [evaluate(arg, env) for arg in expr['arguments']]
            local_env = env.copy()
            local_env.update(zip(params, args))
            return evaluate(func['body'], local_env)
        elif expr['type'] == 'operation':
            op = expr['operator']
            args = [evaluate(arg, env) for arg in expr['arguments']]
            if op == '+':
                return sum(args)
            elif op == '-':
                return args[0] - args[1]
            elif op == '*':
                return args[0] * args[1]
            elif op == '/':
                return args[0] / args[1]
            elif op == '=':
                return args[0] == args[1]
            elif op == '>':
                return args[0] > args[1]
            elif op == '<':
                return args[0] < args[1]
            elif op == '>=':
                return args[0] >= args[1]
            elif op == '<=':
                return args[0] <= args[1]
        elif expr['type'] == 'if_condition':
            test = evaluate(expr['test'], env)
            if test:
                return evaluate(expr['then'], env)
            else:
                return evaluate(expr['else'], env)
    return expr```
