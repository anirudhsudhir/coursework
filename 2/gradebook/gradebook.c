#include "gradebook.h"
#include "studentDetails.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int totalSubs = 0;
int totalInternals = 0;
int totalExternals = 0;
int currentStudentPointer = 0;
int max_internal_marks;
int max_external_marks;

GradeBook **gradeBook;
char **subjects;

int choiceEntry() {
  clear();
  printf("Enter 1 to add a student\n");
  printf("Enter 2 to update all student scores\n");
  printf("Enter 3 to calculate marks data\n");
  printf("Enter 4 to display student data along with marks\n");
  printf("Enter 5 to update a student's score via pattern match\n");
  printf("Enter 6 to quit the application\n");
  int option;
  scanf("%d", &option);

  switch (option) {
  case 1:
    addStudent();
    break;
  case 2:
    if (currentStudentPointer == 0) {
      printf("No student has been added yet\n");
      waitClear();
    } else {
      updateStudentScores();
    }
    break;
  case 3:
    if (currentStudentPointer == 0) {
      printf("No student has been added yet\n");
      waitClear();
    } else {
      calculateStudentScores();
    }
    break;
  case 4:
    if (currentStudentPointer == 0) {
      printf("No student has been added yet\n");
      waitClear();
    } else {
      displayStudentData();
      waitClear();
    }
    break;
  case 5:
    if (currentStudentPointer == 0) {
      printf("No student has been added yet\n");
      waitClear();
    } else {
      updateSingleStudentScores();
      waitClear();
    }
    break;
  case 6:
    writeToFile();
    printf("Data has been written to csv file\n");
    return 1;
  }
  return 0;
}

void readSubjects() {
  for (int i = 0; i < totalSubs; i++) {
    printf("Enter the name of subject %d\n", i + 1);
    subjects[i] = (char *)malloc(1024);
    fgets(subjects[i], 1024, stdin);
    subjects[i][strlen(subjects[i]) - 1] = '\0';
    subjects[i] = realloc(subjects[i], strlen(subjects[i]));
  }
}

void bootstrapGradebook() {
  printf("Enter the total number of subjects\n");
  scanf("%d", &totalSubs);
  subjects = (char **)malloc(totalSubs * sizeof(char *));
  char dummy_space;
  scanf("%c", &dummy_space);
  readSubjects();

  printf("Enter the total number of internals\n");
  scanf("%d", &totalInternals);
  printf("Enter the total number of externals\n");
  scanf("%d", &totalExternals);
  printf("Enter the maximum marks that can be scored in an internal\n");
  scanf("%d", &max_internal_marks);
  printf("Enter the maximum marks that can be scored in an external\n");
  scanf("%d", &max_external_marks);

  printf("Enter 1 to read data from file\n");
  int choice;
  scanf("%d", &choice);
  if (choice == 1) {
    readFromFile();
    waitClear();
  }
}

int main() {
  bootstrapGradebook();

  while (1) {
    if (choiceEntry())
      break;
  }
  freeMemory(gradeBook);
}
