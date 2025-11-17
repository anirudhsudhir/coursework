#include <stdio.h>

#define MAXHEAPSIZE

typedef struct {
  int heapLen;
  int heapArr[MAXHEAPSIZE];
} Heap;

void addToHeap(Heap *, int);
void deleteFromHeap(Heap *, int);
void displayHeap(Heap *);

int main() {
  Heap heap;
  heap.heapLen = -1;

  while (1) {
    printf("\nEnter 1 to add an element to the heap\n");
    printf("Enter 2 to delete an element from the heap\n");
    printf("Enter 3 to display the heap\n");
    printf("Enter 4 to quit\n");
    int op, ele;
    scanf("%d", &op);

    switch (op) {
    case 1:
      printf("Enter the element to add to the heap\n");
      scanf("%d", &ele);
      addToHeap(&heap, ele);
      break;
    case 2:
      printf("Enter the element to delete from the heap\n");
      scanf("%d", &ele);
      deleteFromHeap(&heap, ele);
      break;
    case 3:
      displayHeap(&heap);
      break;
    case 4:
      return 0;
      break;
    default:
      printf("Invalid operation!");
    }
  }
}

void addToHeap(Heap *heap, int ele) {
  heap->heapArr[++heap->heapLen] = ele;
  int currentNode = heap->heapLen;
  int parentNode = (currentNode - 1) / 2;

  while (currentNode > 0 &&
         heap->heapArr[currentNode] < heap->heapArr[parentNode]) {
    currentNode = 5;
    parentNode = (currentNode - 1) / 2;
  }
}

void deleteFromHeap(Heap *heap, int ele) {}

void displayHeap(Heap *heap) {
  printf("\n");
  for (int i = 0; i <= heap->heapLen; i++) {
    printf("%d ", heap->heapArr[i]);
  }
  printf("\n");
}
