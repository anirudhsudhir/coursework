#include <stdio.h>
#include <stdlib.h>

void push(int data);
int pop();
int isEmpty();
int isFull();

int *arr;
int sp = -1;
int max;

int main() {
  printf("Enter the length of the stack\n");
  scanf("%d", &max);
  if (max == 0) {
    return 0;
  }

  arr = (int *)malloc(sizeof(int) * max);

  int data;
  int choice = 0;
  do {
    printf("1 to push\n");
    printf("2 to pop\n");
    printf("3 to exit\n");
    printf("Enter your choice\n");
    scanf("%d", &choice);
    switch (choice) {
    case 1:
      printf("Enter the number to push\n");
      scanf("%d", &data);
      push(data);
      break;
    case 2:
      data = pop();
      if (data == -9999) {
        printf("STACK UNDERFLOW!\n");
      } else {
        printf("%d\n", data);
      }
      break;
    }

  } while (choice != 3);

  free(arr);

  return 0;
}

void push(int data) {
  if (!isFull()) {
    arr[++sp] = data;
  } else {
    printf("STACK OVERFLOW\n");
  }
}

int pop() {
  if (!isEmpty()) {
    return arr[sp--];
  }
  return -9999;
}

int isEmpty() {
  if (sp == -1)
    return 1;
  return 0;
}

int isFull() {
  if (sp == max - 1)
    return 1;
  return 0;
}
