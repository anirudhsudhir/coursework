#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define KEYSIZE 16

int main() {
  int i, j;
  FILE *f;
  char key[KEYSIZE];
  int value1, value2;

  /* use the output of the previous step as value1 and value2
  respectively*/
  value1 = 1524013729;
  value2 = 1524020929;

  f = fopen("keys.txt", "w");
  for (j = value1; j <= value2; j++) {
    srand(j);
    for (i = 0; i < KEYSIZE; i++) {
      key[i] = rand() % 256;
      fprintf(f, "%.2x", (unsigned char)key[i]);
    }
    fprintf(f, "\n");
  }
}
