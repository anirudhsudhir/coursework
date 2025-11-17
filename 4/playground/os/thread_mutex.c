#include <pthread.h>
#include <stdio.h>

#define COUNT 30000

int count = 0;

void *incr(void *mutex) {
  pthread_mutex_lock(mutex);
  count++;
  pthread_mutex_unlock(mutex);

  return NULL;
}

int main() {
  printf("Count is :%d\n", count);

  pthread_t arr[COUNT];
  pthread_mutex_t mutex;
  pthread_mutex_init(&mutex, NULL);

  for (int i = 0; i < COUNT; i++) {
    pthread_create(&arr[i], NULL, incr, &mutex);
  }

  for (int i = 0; i < 30000; i++) {
    pthread_join(arr[i], NULL);
  }

  printf("Count is :%d\n", count);

  return 0;
}
