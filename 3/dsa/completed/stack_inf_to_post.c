#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#define MAX 100

char Stack[MAX];
char Postfix[MAX];
char Infix[MAX];
int infixIndex = 0;
int stackIndex = -1;
int postfixIndex = -1;

void push(char);
char pop();

void convert();
int precedence(char);
char associativity(char);

int evalPostfix();

int main() {
  scanf("%s", Infix);
  printf("\nInfix expression: %s\n\n", Infix);
  convert();
  printf("\nPostfix expression: %s\n", Postfix);
  printf("Result of Postfix evaluation %d\n", evalPostfix());
  exit(EXIT_SUCCESS);
}

void push(char operand) {
  if (stackIndex != MAX - 1) {
    Stack[++stackIndex] = operand;
  } else {
    exit(EXIT_SUCCESS);
  }
}

char pop() {
  if (stackIndex > -1) {
    return Stack[stackIndex--];
  } else {
    return '\0';
  }
}
int precedence(char operator) {
  switch (operator) {
  case '^':
    return 3;
  case '*':
    return 2;
  case '/':
    return 2;
  case '+':
    return 1;
  case '-':
    return 1;
  default:
    printf("Invalid operator precedence check: %c\n", operator);
    exit(EXIT_SUCCESS);
  }
  return 0;
}

char associativity(char operator) {
  switch (operator) {
  case '^':
    return 'R';
  }
  return 'L';
}

void convert() {
  char symbol = Infix[infixIndex];
  while (Infix[infixIndex] != '\0') {
    symbol = Infix[infixIndex];
    if (isdigit(symbol)) {
      Postfix[++postfixIndex] = symbol;
      ++infixIndex;
      printf("Adding digit %c to Postfix\n", symbol);
    } else {

      if (symbol == '(') {
        push(symbol);
        ++infixIndex;
        printf("Adding %c to Stack\n", symbol);
      } else if (symbol == ')') {

        char val = pop();
        while (val != '(') {
          Postfix[++postfixIndex] = val;
          printf("Adding symbol %c to Postfix in (\n", val);
          val = pop();
        }
        ++infixIndex;

      } else {

        if (stackIndex == -1) {
          push(symbol);
          ++infixIndex;
          printf("Adding %c to Stack\n", symbol);

          continue;
        }

        if (Stack[stackIndex] == '(') {
          push(symbol);
          ++infixIndex;
          printf("Adding %c to Stack after (\n", symbol);
          continue;
        }

        if (precedence(symbol) < precedence(Stack[stackIndex]) ||
            (precedence(symbol) == precedence(Stack[stackIndex]) &&
             associativity(symbol) == 'L')) {
          Postfix[++postfixIndex] = pop();
          printf("Adding %c to Postfix in regular push\n",
                 Postfix[postfixIndex]);
        } else {
          push(Infix[infixIndex]);
          ++infixIndex;
          printf("Adding %c to Stack\n", symbol);
        }
      }
    }
  }

  char rem = pop();
  while (rem != '\0') {
    Postfix[++postfixIndex] = rem;
    rem = pop();
  }
}

int evalPostfix() {
  int eval = 0;
  int tempPostfixIndex = 0;
  while (tempPostfixIndex <= postfixIndex) {
    char symbol = Postfix[tempPostfixIndex++];
    if (isdigit(symbol)) {
      push(symbol);
    } else {
      int operand2 = pop() - '0';
      int operand1 = pop() - '0';
      switch (symbol) {
      case '+':
        printf("Adding %d, %d\n", operand1, operand2);
        push((operand1 + operand2) + '0');
        break;
      case '-':
        push((operand1 - operand2) + '0');
        break;
      case '*':
        push((operand1 * operand2) + '0');
        break;
      case '/':
        push((operand1 / operand2) + '0');
        break;
      case '^':
        push((operand1 ^ operand2) + '0');
        break;
      default:
        printf("Invalid symbol: %c\n", symbol);
      }
    }
  }
  return pop() - '0';
}
