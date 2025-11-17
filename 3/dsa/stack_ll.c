#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int data;
  struct Node *next;
} Node;

Node *newNode(int);
void freeStack();

void push(int);
int pop();

Node *head = NULL;

int main() {
  int choice, data;
  do {
    printf("1 to push\n");
    printf("2 to pop\n");
    printf("3 to exit\n");
    printf("Enter your choice\n");
    scanf("%d", &choice);

    switch (choice) {
    case 1:
      printf("Enter value to push\n");
      scanf("%d", &data);
      push(data);
      break;
    case 2:
      data = pop();
      if (data == -9999) {
        printf("STACK UNDERFLOW\n");
      } else {
        printf("Popped : %d\n", data);
      }
      break;
    }
  } while (choice != 3);

  freeStack();
  return 0;
}

Node *newNode(int data) {
  Node *node = (Node *)malloc(sizeof(Node));
  node->data = data;
  node->next = NULL;
  return node;
}

void push(int data) {
  Node *node = newNode(data);
  if (node == NULL) {
    printf("STACK OVERFLOW\n");
    return;
  }
  node->next = head;
  head = node;
}

int pop() {
  if (head == NULL) {
    return -9999;
  }

  int data = head->data;
  head = head->next;
  return data;
}

void freeStack() {
  Node *temp = head;
  while (temp != NULL) {
    Node *temp2 = temp;
    temp = temp->next;
    free(temp2);
  }
}
