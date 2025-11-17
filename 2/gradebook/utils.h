#include "gradebook.h"

extern int totalSubs;
extern int totalInternals;
extern int totalExternals;
extern int maxStudents;
extern int currentStudentPointer;
extern int max_internal_marks;
extern int max_external_marks;

extern char **subjects;
extern GradeBook **gradeBook;

void freeMemory(GradeBook *gradeBook[]);
void writeToFile();
void readFromFile();
