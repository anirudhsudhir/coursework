#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
  int data;
  struct Node *next;
} Node;

typedef struct {
  Node *head;
  Node *tail;
  int size;
} CircularLinkedList;

// Initialize the circular linked list to be empty
void initList(CircularLinkedList *list);
// Insert a node with the given value at the end of the circular linked list
void insertEnd(CircularLinkedList *list, int value);
// Solve the Josephus problem by eliminating every k-th node until only 3 nodes
// remain
void josephus(CircularLinkedList *list, int k);
// Print the information of all the nodes in the circular linked list
void printList(CircularLinkedList *list);
// Destroy the nodes in the circular linked list and reinitialize it to be empty
void destroyList(CircularLinkedList *list);

int main() {
  int n, k;
  CircularLinkedList list;

  initList(&list);

  scanf("%d", &n);
  scanf("%d", &k);

  for (int i = 1; i <= n; i++) {
    insertEnd(&list, i);
  }

  josephus(&list, k);
  printList(&list);
  destroyList(&list);
  return 0;
}

void deleteNode(CircularLinkedList *list, Node *prev, Node *current) {
  // Complete this function
  if (list->size <= 3) {
    printf("Error: Three or lesser nodes remaining!");
    return;
  }

  if (current == list->head) {
    list->head = current->next;
    list->tail->next = list->head;
  } else if (current == list->tail) {
    list->tail = prev;
    prev->next = list->head;
  } else {
    prev->next = current->next;
  }

  list->size--;
  free(current);
}

void josephus(CircularLinkedList *list, int k) {
  // complete this function

  // testing deleteNode()
  // printList(list);
  // deleteNode(list, list->head->next, list->head->next->next);
  // printList(list);
  // deleteNode(list, NULL, list->head);
  // printList(list);

  Node *toDelete = list->head;
  while (list->size > k) {
    for (int i = 0; i < k - 2; i++) {
      toDelete = toDelete->next;
    }

    deleteNode(list, toDelete, toDelete->next);
    toDelete = toDelete->next;
  }
}

void initList(CircularLinkedList *list) {
  list->head = NULL;
  list->tail = NULL;
  list->size = 0;
}

void insertEnd(CircularLinkedList *list, int value) {
  Node *newNode = (Node *)malloc(sizeof(Node));
  newNode->data = value;
  newNode->next = NULL;

  if (list->head == NULL) {
    list->head = newNode;
    list->tail = newNode;
    newNode->next = list->head;
  } else {
    list->tail->next = newNode;
    list->tail = newNode;
    list->tail->next = list->head; // Circular linking
  }
  list->size++;
}

void printList(CircularLinkedList *list) {
  if (list->head == NULL) {
    return;
  }

  Node *current = list->head;
  do {
    printf("%d ", current->data);
    current = current->next;
  } while (current != list->head);

  printf("\n");
}

void destroyList(CircularLinkedList *list) {
  if (list->head != NULL) {
    Node *p = list->head;

    while (p != list->tail) {
      list->head = p->next;
      list->tail->next = p->next;
      //			printf("%d freed\n",p->data);
      free(p);
      p = list->head;
    }
    //		printf("%d freed\n",p->data);
    // free(p);
    list->head = NULL;
    list->tail = NULL;
  }
}
