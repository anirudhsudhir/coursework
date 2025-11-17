#include <stdio.h>

#define MAX 10

void enqueue(int *queue, int ele, int *f, int *r);
int dequeue(int *queue, int *f, int *r);
int timecount(int *q, int ticket_count, int counter_close, int person, int *f,
              int *r);

void enqueue(int *queue, int ele, int *f, int *r) {
  if (*f == -1) {
    *f = 0;
  }

  // if (*r != *f) {
  *r = (*r + 1) % MAX;
  queue[(*r) % MAX] = ele;
  // } else {
  // printf("QUEUE OVERFLOW!");
  // }
}
int dequeue(int *queue, int *f, int *r) {
  // if ((*f != -1) && (*f != (*r) + 1)) {
  int val = queue[(*f) % MAX];
  *f = (*f + 1) % MAX;
  return val;
  // } else {
  //   printf("QUEUE UNDERFLOW!");
  // }
  // return -1;
}

int timecount(int *q, int ticket_count, int counter_close, int person, int *f,
              int *r) {

  int track_person = person;
  int delay = 0;
  int time = 0;

  while (q[track_person] > 0) {
    if (delay == ticket_count) {
      delay = 0;
      time += counter_close;
    }
    int old_r = *r;

    int val = dequeue(q, f, r);
    printf("dequeing %d\n", val);

    if (val > 0) {

      enqueue(q, val - 1, f, r);

      int new_r = *r;
      if (old_r == track_person)
        track_person += new_r - old_r;
      printf("enquing %d\n, track_person = %d", val - 1, track_person);
    }
    delay++;
    time++;

    for (int i = 0; i < MAX; i++) {
      printf("%d ", q[i]);
    }
    printf("\n");
  }

  return time;
}

int main() {
  // Your code here
  int q[MAX];
  int f = -1;
  int r = -1;
  int ticket_count, counter_close, person, num_persons;

  scanf("%d", &num_persons);
  int tickets_per_person[num_persons];

  for (int i = 0; i < num_persons; i++) {
    scanf("%d", &tickets_per_person[i]);
  }

  scanf("%d", &ticket_count);
  scanf("%d", &counter_close);
  scanf("%d", &person);

  for (int i = 0; i < num_persons; i++) {
    enqueue(q, tickets_per_person[i], &f, &r);
  }

  // for (int i = 0; i < num_persons; i++) {
  //   printf("%d ", q[i]);
  // }

  int time = timecount(q, ticket_count, counter_close, person, &f, &r);
  printf("%d\n", time);

  return 0;
}
