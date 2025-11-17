#include "utils.h"
#include "gradebook.h"
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void freeMemory(GradeBook *gradeBook[]) {
  for (int i = 0; i < currentStudentPointer; i++) {
    if (gradeBook[i] != NULL) {

      if (gradeBook[i]->marks != NULL) {
        for (int j = 0; j < totalSubs; j++) {
          if (gradeBook[i]->marks[j] != NULL)
            free(gradeBook[i]->marks[j]);
        }
        free(gradeBook[i]->marks);
      }

      if (gradeBook[i]->result != NULL) {
        if (gradeBook[i]->result->subjectGPA != NULL) {
          free(gradeBook[i]->result->subjectGPA);
        }
        free(gradeBook[i]->result);
      }

      free(gradeBook[i]->name);
      free(gradeBook[i]);
    }
  }

  free(gradeBook);

  for (int i = 0; i < totalSubs; i++) {
    free(subjects[i]);
  }

  free(subjects);
}

void writeToFile() {
  FILE *file = fopen("StudentData.csv", "w");
  if (file != NULL) {
    fprintf(file, "%s,", "Name");

    for (int j = 0; j < totalSubs; j++) {
      for (int k = 0; k < totalInternals; k++) {
        fprintf(file, "%s-Internal-%d,", subjects[j], k + 1);
      }
      for (int k = 0; k < totalExternals; k++) {
        fprintf(file, "%s-External-%d,", subjects[j], k + 1);
      }
    }

    fprintf(file, "%s,", "Roll No");
    fprintf(file, "%s,", "Section");
    fprintf(file, "%s,", "Semester");
    fprintf(file, "%s,", "Phone Number");

    for (int j = 0; j < totalSubs; j++) {
      fprintf(file, "%s-Percentage,", subjects[j]);
    }
    fprintf(file, "%s,;\n", "GPA");

    for (int i = 0; i < currentStudentPointer; i++) {
      fwrite(gradeBook[i]->name, 1, sizeof(gradeBook[i]->name), file);
      putc(',', file);

      for (int j = 0; j < totalSubs; j++) {
        for (int k = 0; k < totalInternals + totalExternals; k++) {
          fprintf(file, "%d,", gradeBook[i]->marks[j][k]);
        }
      }

      fprintf(file, "%d,", gradeBook[i]->roll_no);
      fputc(gradeBook[i]->section, file);
      fputc(',', file);
      fprintf(file, "%d,", gradeBook[i]->sem);
      fprintf(file, "%ld,", gradeBook[i]->phone_no);

      for (int j = 0; j < totalSubs; j++) {
        fprintf(file, "%f,", gradeBook[i]->result->subjectGPA[j]);
      }
      fprintf(file, "%f,;\n", gradeBook[i]->result->GPA);
    }
    fclose(file);
  } else {
    perror("ERROR: ");
  }
}

void readFromFile() {
  FILE *file = fopen("StudentData.csv", "r");
  if (file != NULL) {
    // Skipping first line containg headings
    char ch = fgetc(file);
    while (ch != ';') {
      ch = fgetc(file);
    }

    char *line = (char *)calloc(100, sizeof(char));
    while (1) {
      int counter = 0;

      ch = fgetc(file);
      while (ch != ';' && ch != EOF) {
        if (isalnum(ch) || ch == ',' || ch == '.' || ch == ' ') {
          line[counter++] = ch;
        }
        ch = fgetc(file);
      }
      line[counter] = '\0';

      if (ch == EOF && sizeof(line) < 2) {
        break;
      }

      char *splitLine = strtok(line, ",");
      while (splitLine != NULL) {
        GradeBook *student = (GradeBook *)malloc(sizeof(GradeBook));

        student->name = (char *)malloc(1024);
        memcpy(student->name, splitLine, strlen(splitLine) + 1);
        student->name = realloc(student->name, strlen(student->name) + 1);

        splitLine = strtok(NULL, ",");

        int **marks = (int **)malloc(sizeof(int *) * totalSubs);
        for (int i = 0; i < totalSubs; i++) {
          int *subMarks =
              (int *)malloc(sizeof(int) * (totalInternals + totalExternals));

          for (int j = 0; j < totalInternals; j++) {
            subMarks[j] = atoi(splitLine);
            splitLine = strtok(NULL, ",");
          }
          for (int j = totalInternals; j < totalInternals + totalExternals;
               j++) {
            subMarks[j] = atoi(splitLine);
            splitLine = strtok(NULL, ",");
          }

          marks[i] = subMarks;
        }

        student->marks = marks;

        student->roll_no = atoi(splitLine);
        splitLine = strtok(NULL, ",");
        student->section = splitLine[0];
        splitLine = strtok(NULL, ",");
        student->sem = atoi(splitLine);
        splitLine = strtok(NULL, ",");
        student->phone_no = atoi(splitLine);
        splitLine = strtok(NULL, ",");

        Result *result = (Result *)malloc(sizeof(Result));
        float *calculatedSubjectPercent =
            (float *)malloc(totalSubs * sizeof(float));
        for (int j = 0; j < totalSubs; j++) {
          calculatedSubjectPercent[j] = atof(splitLine);
          splitLine = strtok(NULL, ",");
        }
        result->GPA = atof(splitLine);
        splitLine = strtok(NULL, ",");
        result->subjectGPA = calculatedSubjectPercent;

        student->result = result;

        if (gradeBook != NULL) {
          gradeBook = (GradeBook **)realloc(
              gradeBook, (currentStudentPointer + 1) * sizeof(GradeBook *));
          gradeBook[currentStudentPointer++] = student;
        } else {
          currentStudentPointer = 0;
          gradeBook = (GradeBook **)malloc(sizeof(GradeBook *));
          gradeBook[currentStudentPointer++] = student;
        }
      }

      counter = 0;

      if (ch == EOF) {
        break;
      }
    }

    free(line);
    fclose(file);
  } else {
    perror("ERROR: ");
  }
}
