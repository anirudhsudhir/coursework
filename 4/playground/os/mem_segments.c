#include <stdio.h>

int GLOBAL_CONST = 5;
int GLOBAL_UNDEFINED;

int main() {
  int local_var = 10;
  printf("Printing local_var = %d", local_var);
  printf("Printing global const = %d", GLOBAL_CONST);
  GLOBAL_UNDEFINED = 15;
  printf("Printing global undefined = %d", GLOBAL_UNDEFINED);
}
