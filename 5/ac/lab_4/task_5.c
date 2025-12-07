#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define KEY_SIZE 16

void generate_key(uint8_t *key) {
  int random_fd = open("/dev/urandom", O_RDONLY);
  if (random_fd < 0) {
    exit(1);
  }

  ssize_t result = read(random_fd, key, KEY_SIZE);
  if (result != KEY_SIZE) {
    close(random_fd);
    exit(1);
  }

  close(random_fd);
}

void print_key(uint8_t *key) {
  for (int i = 0; i < KEY_SIZE; i++) {
    printf("%x", key[i]);
  }
  printf("\n");
}

int main() {
  uint8_t key[KEY_SIZE];

  generate_key(key);
  print_key(key);
  return 0;
}
