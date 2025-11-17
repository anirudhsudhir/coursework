.global _start

.section .data
a: .word 1,2,3,4,5,6,7,8,9
b: .word 1,2,3,4,5,6,7,8,9
c: .word 0,0,0,0,0,0,0,0,0

.section .text

_start:

  ldr r1, =a
  ldr r2, =b
  ldr r3, =c

  mov r4, #0 ; i

i_loop: cmp r4, #3 ; i loop invariant
  beq fin

  mov r5, #0 ; j
  bl j_loop

  add r4, r4, #1
  b i_loop

fin: swi 0x011

j_loop: cmp r5, #0
  stmeqfd sp!, {lr}

  cmp r5, #3 ; j loop invariant

  ldmeqfd sp!, {lr}
  bxeq lr

  mov r6, #0 ; k
  bl k_loop

  add r5, r5, #1
  b j_loop

k_loop: cmp r6, #3; k loop invariant
  bxeq lr

  mov r9, #3
  mla r7, r4, r9, r6
  mov r9, #4
  mla r7, r9, r7, r1

  mov r9, #3
  mla r8, r6, r9, r5
  mov r9, #4
  mla r8, r9, r8, r2

  ldr r7, [r7]
  ldr r8, [r8]
  mul r7, r8, r7

  mov r9, #3
  mla r8, r4, r9, r5
  mov r9, #4
  mla r8, r9, r8, r3

  ldr r9, [r8]
  add r9, r9, r7
  str r9, [r8]

  add r6, r6, #1
  b k_loop

