#include <stdio.h>
#include <stdlib.h>

#define MAX_QUEUE_SIZE 100

typedef struct {
  int data[MAX_QUEUE_SIZE];
  int front;
  int rear;
  int size;
} Queue;

void initializeQueue(Queue *queue) {
  // initilaize the queue
  queue->front = -1;
  queue->rear = -1;
  queue->size = MAX_QUEUE_SIZE;
}

void enqueue(Queue *queue, int person) {
  // complete this function to perform enqueue function , also if queue is full
  // print "queue is full"

  if (queue->front == -1) {
    queue->front = 0;
  }

  if (!(queue->rear == MAX_QUEUE_SIZE - 1)) {
    queue->data[++queue->rear] = person;
    // printf("enq %d\n",queue->rear);
  } else {
    printf("queue is full\n");
  }
}

int isQueueEmpty(Queue *queue) {
  // check if the queue is empty
  if ((queue->rear == -1) || (queue->front == queue->rear + 1)) {
    return 1;
  }
  return 0;
}

int dequeue(Queue *queue) {
  // function to dequeue element from queue
  if (!isQueueEmpty(queue)) {
    // printf("deq %d",queue->front);
    return queue->data[queue->front++];
  } else {
    // Queue underflow
    return -1;
  }
}

Queue *createQueue() {
  Queue *queue = (Queue *)malloc(sizeof(Queue));
  if (queue == NULL) {

    exit(EXIT_FAILURE);
  }
  initializeQueue(queue);
  return queue;
}

void roundRobinDequeue(Queue **queues, int n) {
  // complete this function to perform dequeue operation in round robin fashion.
  int q_empty = 0;
  do {
    q_empty = 0;
    for (int i = 0; i < n; i++) {
      int person = dequeue(queues[i]);

      if (person != -1) {
        printf("%d ", person);
      } else {
        // printf("%d ", person);
        q_empty++;
      }
    }
  } while (q_empty < n);
}

int main() {
  int m, n;

  // Read the total number of people and number of queues
  scanf("%d", &m);
  scanf("%d", &n);

  // Read the people array
  int *arr = (int *)malloc(m * sizeof(int));
  for (int i = 0; i < m; ++i) {
    scanf("%d", &arr[i]);
  }

  // Create and initialize queues
  Queue **queues = (Queue **)malloc(n * sizeof(Queue *));
  for (int i = 0; i < n; ++i) {
    queues[i] = createQueue();
  }

  // Distribute people into queues
  int index = 0;
  for (int i = 0; i < m; ++i) {
    enqueue(queues[index], arr[i]);
    if ((i + 1) % (m / n) == 0 && index < n - 1) {
      index++;
    }
  }

  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < MAX_QUEUE_SIZE; j++)
  //     printf("%d ", queues[i]->data[j]);
  //   printf("\n");
  // }

  // Perform the round-robin dequeuing
  roundRobinDequeue(queues, n);

  // Free memory
  free(arr);
  for (int i = 0; i < n; ++i) {
    free(queues[i]);
  }
  free(queues);

  return 0;
}
