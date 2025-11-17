CMAKE_MINIMUM_REQUIRED(VERSION 3.31.0)

# here we specify that the project is C language only, so the default
# C compiler on the system will be used
PROJECT(os-playground C)

ADD_EXECUTABLE(signals signals.c)
ADD_EXECUTABLE(mem_segments mem_segments.c)
ADD_EXECUTABLE(threads threads.c)
