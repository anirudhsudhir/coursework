#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

void lineage(pid_t pid) {
  pid_t parent = getppid();
  printf("PID: %d\n", pid);

  while (parent != pid && parent != 1) {
    printf("PID: %d\n", parent);
    pid = parent;
    parent = getppid();
  }

  if (parent == 1) {
    printf("init PID: %d\n", parent);
  }
}

int main() {
  pid_t pid = getpid();
  pid_t child_pid = fork();

  if (child_pid == 0) {
    printf("child process");
    lineage(getpid());
    exit(0);
  } else if (child_pid > 0) {
    lineage(getpid());
  } else {
    printf("failed to fork()");
    exit(1);
  }

  return 0;
}
