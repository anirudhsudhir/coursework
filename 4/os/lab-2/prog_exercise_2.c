#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

int *arr;
int n;
int avg;

int calc_avg = 0;

pthread_mutex_t m1;
pthread_cond_t c1;
pthread_mutex_t c1m;

void *t1_work() {
  pthread_mutex_lock(&m1);
  printf("Thread 1:\n");
  printf("Enter the number of elements in the array:\n");
  scanf("%d", &n);
  arr = (int *)malloc(n * sizeof(int));

  printf("Enter the elements of the array\n");
  for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
  }

  pthread_mutex_unlock(&m1);

  pthread_mutex_lock(&c1m);
  while (calc_avg == 0)
    pthread_cond_wait(&c1, &c1m);
  pthread_mutex_unlock(&c1m);

  pthread_mutex_lock(&m1);

  printf("Thread 1:\n");
  printf("Average of integers = %d\n", avg);

  pthread_mutex_unlock(&m1);

  return NULL;
}

void *t2_work() {
  pthread_mutex_lock(&m1);

  printf("Thread 2: Calculating the average\n");
  int sum = 0;
  for (int i = 0; i < n; i++)
    sum += arr[i];
  avg = sum / n;

  pthread_mutex_lock(&c1m);
  calc_avg = 1;
  pthread_cond_signal(&c1);
  pthread_mutex_unlock(&c1m);

  pthread_mutex_unlock(&m1);

  return NULL;
}

int main() {
  pthread_t t1;
  pthread_t t2;

  pthread_mutex_init(&m1, NULL);

  pthread_cond_init(&c1, NULL);
  pthread_mutex_init(&c1m, NULL);

  pthread_create(&t1, NULL, t1_work, NULL);
  pthread_create(&t2, NULL, t2_work, NULL);

  pthread_join(t1, NULL);
  pthread_join(t2, NULL);
  return 0;
}
