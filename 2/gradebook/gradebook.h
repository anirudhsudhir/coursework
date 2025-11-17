#pragma once

typedef struct {
  float *subjectGPA;
  float GPA;
} Result;

typedef struct {
  char *name;
  int **marks;
  int roll_no;
  char section;
  int sem;
  long int phone_no;
  Result *result;
} GradeBook;

#define clear() printf("\x1b[H\x1b[J");
#define waitClear()                                                            \
  printf("Enter any character to continue\n");                                 \
  getchar();                                                                   \
  getchar();
#define screenSep()                                                            \
  for (int i = 0; i < 75; i++) {                                               \
    printf("=");                                                               \
  }                                                                            \
  printf("\n");

int main();

void readSubjects();

void bootstrapGradebook();

int choiceEntry();
