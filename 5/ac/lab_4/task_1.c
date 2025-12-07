#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KEY_SIZE 16

void generate_key(uint8_t *key) {
  for (int i = 0; i < KEY_SIZE; i++) {
    key[i] = rand() % 256;
  }
}

void print_key(uint8_t *key) {
  printf("Time-based seed: ");
  for (int i = 0; i < KEY_SIZE; i++) {
    printf("%x", key[i]);
  }
  printf("\n");
}

int main() {
  uint8_t key[KEY_SIZE];
  srand(time(NULL));
  generate_key(key);
  print_key(key);
  return 0;
}
