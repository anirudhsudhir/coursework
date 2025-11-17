#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

int main() {
  printf("Current process -> %d\n", getpid());
  pid_t p1;
  p1 = fork();
  int status;

  if (p1 == -1) {
    perror("Error while forking");
  } else if (p1 == 0) {
    printf("Child process -> %d\n", getpid());
    // child process
    execl("/bin/ls", "ls", "-l", NULL);
    perror("execl failed");
  } else {
    wait(&status);

    if (WIFEXITED(status)) {
      printf("child process terminated\n");
    } else {
      printf("child process terminated abnormally\n");
    }
  }

  return 0;
}
