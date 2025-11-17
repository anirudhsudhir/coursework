#include <pthread.h>
#include <semaphore.h>

void *read(void *ptr) { return NULL; }

int main() {
  pthread_t t1;
  sem_t s1;

  sem_open("SEM1", s1, 0666, NULL);
  pthread_create(&t1, NULL, read, &s1);
  pthread_join(t1, NULL);
  return 0;
}
