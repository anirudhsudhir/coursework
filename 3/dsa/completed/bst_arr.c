#include <stdio.h>

#define CAP 10

typedef struct {
  int key;
  int val;
  int flag;
} Node;

void insertNode(Node *, int, int, int);
void preorderTraversal(Node *, int);
void deleteNode(Node *, int, int);

int inorderSuccessor(Node *, int, int);

int i = 0;

int main() {
  // int arr[] = {1, 2, 3, 4};
  Node arr[CAP];

  insertNode(arr, 10, 'a', -1);
  insertNode(arr, 8, 'b', i);
  insertNode(arr, 9, 'd', i);
  insertNode(arr, 7, 'c', i);
  insertNode(arr, 12, 'e', i);
  insertNode(arr, 13, 'f', i);
  insertNode(arr, 11, 'g', i);

  preorderTraversal(arr, 0);
  printf("\n");
}

void preorderTraversal(Node *arr, int index) {
  if (index < CAP && arr[index].flag == 1) {
    printf("key %d -> val %c, ", arr[index].key, arr[index].val);
    preorderTraversal(arr, 2 * index + 1);
    preorderTraversal(arr, 2 * index + 2);
  }
}

void insertNode(Node *arr, int key, int val, int index) {
  if (index > CAP - 1) {
    printf("BST is full, cannot add nodes!");
    return;
  }

  if (index == -1) {
    arr[++index].key = key;
    arr[index].val = val;
    arr[index].flag = 1;
    return;
  }
  if (arr[index].flag != 1) {
    arr[index].key = key;
    arr[index].val = val;
    arr[index].flag = 1;
  } else if (arr[index].key > key) {
    insertNode(arr, key, val, 2 * index + 1);
  } else if (arr[index].key < key) {
    insertNode(arr, key, val, 2 * index + 2);
  }
}

// wip
void deleteNode(Node *arr, int key, int index) {
  if (index > CAP - 1) {
    printf("Cannot delete node at index %d", index);
    return;
  }

  if (arr[index].flag == 1 && arr[index].key == key) {
    // delete node logic

    if (2 * index + 1 > CAP - 1 || arr[2 * index + 1].flag != 1) {
    }

    int inorderSuccessorVal = inorderSuccessor(arr, key, 2 * index + 2);
    arr[index].key = arr[inorderSuccessorVal].key;
    arr[index].val = arr[inorderSuccessorVal].val;
    deleteNode(arr, key, 2 * index + 2);
  } else if (arr[index].flag == 1 && arr[index].key < key) {
    deleteNode(arr, key, 2 * index + 2);
  } else if (arr[index].flag == 1 && arr[index].key > key) {
    deleteNode(arr, key, 2 * index + 1);
  }
}

int inorderSuccessor(Node *arr, int key, int index) {
  while (arr[2 * index + 1].flag == 1 && index < CAP - 1) {
    index = 2 * index + 1;
  }
  return index;
}
