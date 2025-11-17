#include <stdio.h>

void print_arr(int[], int);
void quicksort(int[], int);

int main() {
  int arr[] = {5, 3, 89, 45, 78, 90, 106, 22, 71, 2};
  int len = sizeof(arr) / sizeof(arr[0]);

  printf("Before sorting\n");
  print_arr(arr, len);

  quicksort(arr, len);

  printf("After sorting\n");
  print_arr(arr, len);
  return 0;
}

void print_arr(int arr[], int len) {
  for (int i = 0; i < len; i++)
    printf("%d ", arr[i]);
  printf("\n");
}

void quicksort(int arr[], int len) {}
