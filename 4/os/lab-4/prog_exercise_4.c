#include <glob.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Usage: ./prog_exercise_4 [directory and filename]\n");
    return 1;
  }

  glob_t gl;
  gl.gl_offs = 2;
  int match = glob(argv[1], GLOB_DOOFFS, NULL, &gl);
  if (match != 0) {
    printf("no matches found!\n");
    return 1;
  }

  for (int i = 0; i < gl.gl_pathc + gl.gl_offs + 1; i++) {
    if (gl.gl_pathv[i] != NULL)
      printf("%s\n", gl.gl_pathv[i]);
  }

  globfree(&gl);
  return 0;
}
