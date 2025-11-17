#include "studentDetails.h"
#include "gradebook.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void addStudent() {
  GradeBook *student = (GradeBook *)malloc(sizeof(GradeBook));
  student->name = (char *)malloc(1024);
  printf("Enter the name of the student\n");
  char dummy_space;
  scanf("%c", &dummy_space);
  fgets(student->name, 1024, stdin);
  student->name[strlen(student->name) - 1] = '\0';
  student->name = realloc(student->name, strlen(student->name) + 1);
  printf("Enter the roll number of the student\n");
  scanf("%d", &student->roll_no);
  printf("Enter the section of the student\n");
  scanf("%c", &dummy_space);
  scanf("%c", &student->section);
  printf("Enter the semester the student is in\n");
  scanf("%d", &student->sem);
  printf("Enter the phone number of the student\n");
  scanf("%ld", &student->phone_no);

  student->marks = NULL;
  student->result = NULL;

  gradeBook = (GradeBook **)realloc(gradeBook, (currentStudentPointer + 1) *
                                                   sizeof(GradeBook *));
  gradeBook[currentStudentPointer++] = student;
}

void updateStudentScores() {
  for (int h = 0; h < currentStudentPointer; h++) {
    int **marks = (int **)malloc(sizeof(int *) * totalSubs);
    screenSep();
    for (int i = 0; i < totalSubs; i++) {
      printf("Enter %s's marks in %s\n", gradeBook[h]->name, subjects[i]);
      int *subMarks =
          (int *)malloc(sizeof(int) * (totalInternals + totalExternals));

      for (int j = 0; j < totalInternals; j++) {
        printf("Internal %d: ", j + 1);
        scanf("%d", &subMarks[j]);
      }
      for (int j = totalInternals; j < totalInternals + totalExternals; j++) {
        printf("External %d: ", j + 1 - totalInternals);
        scanf("%d", &subMarks[j]);
      }

      marks[i] = subMarks;
    }

    gradeBook[h]->marks = marks;
  }
}

void updateSingleStudentScores() {
  char *nameSearch = (char *)malloc(1024);
  printf("Enter the name of the student whose scores are to be updated\n");
  char dummy_space;
  scanf("%c", &dummy_space);
  fgets(nameSearch, 1024, stdin);
  nameSearch[strlen(nameSearch) - 1] = '\0';

  for (int i = 0; i < strlen(nameSearch); i++) {
    nameSearch[i] = tolower(nameSearch[i]);
  }
  char *lowerName = (char *)malloc(1024);

  for (int i = 0; i < currentStudentPointer; i++) {
    for (int j = 0; j < strlen(gradeBook[i]->name); j++) {
      lowerName[j] = tolower(gradeBook[i]->name[j]);
    }
    lowerName[strlen(gradeBook[i]->name)] = '\0';

    if (strcmp(nameSearch, lowerName) == 0) {
      printf("Enter 1 to update the marks of the following student: %s\n",
             gradeBook[i]->name);
      int choice;
      scanf("%d", &choice);
      if (choice == 1) {

        int **marks = (int **)malloc(sizeof(int *) * totalSubs);
        for (int k = 0; k < totalSubs; k++) {
          printf("Enter %s's marks in %s\n", gradeBook[i]->name, subjects[k]);
          int *subMarks =
              (int *)malloc(sizeof(int) * (totalInternals + totalExternals));

          for (int l = 0; l < totalInternals; l++) {
            printf("Internal %d: ", l + 1);
            scanf("%d", &subMarks[l]);
          }
          for (int l = totalInternals; l < totalInternals + totalExternals;
               l++) {
            printf("External %d: ", l + 1 - totalInternals);
            scanf("%d", &subMarks[l]);
          }
          marks[k] = subMarks;
        }

        if (gradeBook[i]->marks != NULL) {
          for (int j = 0; j < totalSubs; j++) {
            if (gradeBook[i]->marks[j] != NULL)
              free(gradeBook[i]->marks[j]);
          }
          free(gradeBook[i]->marks);
        }

        gradeBook[i]->marks = marks;

        free(lowerName);
        free(nameSearch);
        return;
      }
    }
  }

  free(lowerName);
  free(nameSearch);
  printf("No match found\n");
}

void calculateStudentScores() {
  for (int i = 0; i < currentStudentPointer; i++) {
    if (gradeBook[i]->marks == NULL) {
      printf("Scores cannot be calculated as marks have not been added yet\n");
      waitClear();
      return;
    }
    Result *result = (Result *)malloc(sizeof(Result));
    float *calculatedSubjectPercent =
        (float *)malloc(totalSubs * sizeof(float));
    int MAX_SCORE = totalInternals * max_internal_marks +
                    totalExternals * max_external_marks;
    int calculatedTotalScore = 0;

    for (int j = 0; j < totalSubs; j++) {
      int calculatedSubScore = 0;
      for (int k = 0; k < totalInternals + totalExternals; k++) {
        calculatedSubScore += gradeBook[i]->marks[j][k];
        calculatedTotalScore += gradeBook[i]->marks[j][k];
      }
      calculatedSubjectPercent[j] =
          ((float)calculatedSubScore / MAX_SCORE) * 100;
    }

    result->GPA = ((float)calculatedTotalScore / (MAX_SCORE * totalSubs)) * 10;
    result->subjectGPA = calculatedSubjectPercent;

    if (gradeBook[i]->result != NULL) {
      if (gradeBook[i]->result->subjectGPA != NULL) {
        free(gradeBook[i]->result->subjectGPA);
      }
      free(gradeBook[i]->result);
    }
    gradeBook[i]->result = result;
  }
}

void displayStudentData() {
  screenSep();
  printf("%-7s | %-15s | %-8s | %-8s | %-13s | %-8s\n", "Sl. No.", "Name",
         "Roll No", "Semester", "Phone Number", "Section");
  for (int i = 0; i < currentStudentPointer; i++) {
    screenSep();
    printf("%-7d | %-15s | %-8d | %-8d | %-13ld | %c\n", i + 1,
           gradeBook[i]->name, gradeBook[i]->roll_no, gradeBook[i]->sem,
           gradeBook[i]->phone_no, gradeBook[i]->section);
    if (gradeBook[i]->marks != NULL) {
      for (int j = 0; j < totalSubs; j++) {
        printf("%s Scores\n", subjects[j]);
        for (int k = 0; k < totalInternals; k++) {
          printf("Internal %d - %d\n", k + 1, gradeBook[i]->marks[j][k]);
        }
        for (int k = totalInternals; k < totalInternals + totalExternals; k++) {
          printf("External %d - %d\n", k + 1, gradeBook[i]->marks[j][k]);
        }
      }
      if (gradeBook[i]->result != NULL) {
        for (int k = 0; k < totalSubs; k++) {
          printf("Percentage in %s is %f%%\n", subjects[k],
                 gradeBook[i]->result->subjectGPA[k]);
        }
        printf("%s's GPA is %f\n", gradeBook[i]->name,
               gradeBook[i]->result->GPA);
      } else {
        printf("The GPA of this student has not been calculated\n");
      }
    } else {
      printf("Scores of this student have not been added\n");
    }
  }
}
