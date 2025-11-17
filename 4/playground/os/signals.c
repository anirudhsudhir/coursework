#include <signal.h>
#include <stdio.h>
#include <unistd.h>

void signalHandler(int sig) {
  printf("Handling SIGINT");
  (void)signal(sig, SIG_DFL);
}

int main() {
  (void)signal(SIGINT, signalHandler);

  printf("PID: %d", getpid());
  for (int i = 0;; i++) {
    printf("%d\n", i);
    sleep(1);
  }
  return 0;
}
