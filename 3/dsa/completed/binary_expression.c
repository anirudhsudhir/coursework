#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  char val;
  struct Node *right;
  struct Node *left;
  struct Node *next;
} Node;

Node *newNode(char);
void push(Node **, Node *);
Node *pop(Node **);

int eval(Node *);
void printInorder(Node *);
void printPostorder(Node *);

int main() {
  char expn[] = {'2', '3', '1', '*', '+', '9', '-'};
  int expnSize = sizeof(expn) / sizeof(expn[0]);

  Node *top = NULL;
  for (int i = 0; i < expnSize; i++) {
    char val = expn[i];
    if (val == '+' || val == '-' || val == '*' || val == '/' || val == '^') {
      Node *new = newNode(val);
      Node *r = pop(&top);
      new->left = pop(&top);
      new->right = r;
      push(&top, new);
    } else {
      Node *new = newNode(val);
      push(&top, new);
    }
  }

  printf("\nInorder: ");
  printInorder(top);
  printf("\nPreorder: ");
  printPostorder(top);
  printf("\nEval: %d\n", eval(top));
}

void push(Node **top, Node *node) {
  node->next = *top;
  *top = node;
}

Node *pop(Node **top) {
  Node *node = *top;
  if (node != NULL) {
    *top = (*top)->next;
    return node;
  }

  return NULL;
}

Node *newNode(char val) {
  Node *node = (Node *)malloc(sizeof(Node));
  node->val = val;
  node->right = NULL;
  node->left = NULL;
  node->next = NULL;

  return node;
}

void printInorder(Node *head) {
  if (head == NULL) {
    return;
  }
  printInorder(head->left);
  printf("%c ", head->val);
  printInorder(head->right);
}

void printPostorder(Node *head) {
  if (head == NULL) {
    return;
  }
  printPostorder(head->left);
  printPostorder(head->right);
  printf("%c ", head->val);
}

int eval(Node *head) {
  if (head->left == NULL && head->right == NULL) {
    return (head->val - 48);
  }

  int l = eval(head->left);
  int r = eval(head->right);

  if (head->val == '+') {
    return l + r;
  } else if (head->val == '-') {
    return l - r;
  } else if (head->val == '*') {
    return l * r;
  }

  return l / r;
}
