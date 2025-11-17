#include <stdio.h>
#include <stdlib.h>

struct Node {
  int data;
  struct Node *next;
  struct Node *prev;
};

typedef struct Node Node;
struct Node *createNode(int);
void printList(struct Node *);
void printListReverse(struct Node *);
void freeList(struct Node *);

struct Node *insertBegin(struct Node *, int);
struct Node *insertPos(struct Node *, int, int);
struct Node *insertEnd(struct Node *, int);

struct Node *deleteBegin(struct Node *);
struct Node *deletePos(struct Node *, int);
struct Node *deleteEnd(struct Node *);

int main() {
  // Node *head = createNode(1);
  // Node *two = createNode(2);
  // Node *three = createNode(3);
  //
  // head->next = two;
  // two->next = three;
  // two->prev = head;
  // three->prev = two;

  Node *head = insertBegin(NULL, 3);
  head = insertBegin(head, 2);
  head = insertBegin(head, 1);

  printList(head);
  printListReverse(head);

  head = insertPos(head, 2, 10);
  head = insertPos(head, 5, 98);
  head = insertEnd(head, 100);
  head = insertEnd(head, 101);

  printList(head);
  printListReverse(head);

  head = deleteBegin(head);
  head = deletePos(head, 1);
  head = deletePos(head, 5);
  head = deletePos(head, 2);
  head = deleteEnd(head);

  printList(head);
  printListReverse(head);

  freeList(head);
}

struct Node *createNode(int data) {
  struct Node *node = (struct Node *)malloc(sizeof(struct Node));
  node->data = data;
  node->prev = NULL;
  node->next = NULL;

  return node;
}

void printList(struct Node *head) {
  if (head == NULL) {
    return;
  }

  while (head != NULL) {
    printf("%d ", head->data);
    head = head->next;
  }
  printf("\n");
}

void printListReverse(struct Node *head) {
  if (head == NULL) {
    return;
  }

  while (head->next != NULL) {
    head = head->next;
  }

  while (head->prev != NULL) {
    printf("%d ", head->data);
    head = head->prev;
  }
  printf("%d ", head->data);

  printf("\n");
}

void freeList(struct Node *head) {
  if (head == NULL) {
    return;
  }
  if (head->next != NULL) {
    freeList(head->next);
  }

  free(head);
}

struct Node *insertBegin(struct Node *head, int data) {
  if (head == NULL) {
    return createNode(data);
  }

  Node *node = createNode(data);
  node->next = head;
  head->prev = node;
  head = node;

  return head;
}

struct Node *insertPos(struct Node *head, int pos, int data) {
  if (head == NULL) {
    return createNode(data);
  }

  Node *head_tmp = head;
  for (int i = 1; i < pos - 1; i++) {
    if (head != NULL) {
      head = head->next;
    }
  }

  Node *temp = createNode(data);
  if (head->next != NULL) {
    temp->next = head->next;
    head->next->prev = temp;
  }

  temp->prev = head;
  head->next = temp;
  return head_tmp;
}

struct Node *insertEnd(struct Node *head, int data) {
  if (head == NULL) {
    return createNode(data);
  }

  Node *head_tmp = head;
  while (head_tmp->next != NULL) {
    head_tmp = head_tmp->next;
  }

  Node *temp = createNode(data);
  temp->prev = head_tmp;
  head_tmp->next = temp;

  return head;
}

struct Node *deleteBegin(struct Node *head) {
  if (head == NULL) {
    return NULL;
  }
  if (head->next == NULL) {
    free(head);
    return NULL;
  }

  head = head->next;
  free(head->prev);
  head->prev = NULL;

  return head;
}

struct Node *deletePos(struct Node *head, int pos) {
  while (head == NULL) {
    return NULL;
  }

  if (pos == 1) {
    if (head->next != NULL) {
      head = head->next;
      free(head->prev);
      head->prev = NULL;
    } else {
      free(head);
      return NULL;
    }
    return head;
  }

  Node *temp = head;
  for (int i = 1; i < pos - 1; i++) {
    if (temp == NULL) {
      break;
    }
    temp = temp->next;
  }

  if (temp->next != NULL) {
    if (temp->next->next != NULL) {
      temp->next = temp->next->next;
      free(temp->next->prev);
      temp->next->prev = temp;
    } else {
      free(temp->next);
      temp->next = NULL;
    }
  }

  return head;
}

struct Node *deleteEnd(struct Node *head) {
  if (head == NULL) {
    return NULL;
  }
  if (head->next == NULL) {
    free(head);
    return NULL;
  }

  Node *temp = head;

  while (temp->next->next != NULL) {
    temp = temp->next;
  }

  free(temp->next);
  temp->next = NULL;

  return head;
}
