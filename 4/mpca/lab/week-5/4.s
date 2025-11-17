  .data
a: .asciz "hello world"

  .text
  ldr r1, =a
  mov r2, #0 ; holds len

  add r3, r1, r2
  ldrb r4, [r3]

len_loop: cmp r4, #0
  beq swap

  add r2, r2, #1
  add r3, r1, r2
  ldrb r4, [r3]

  b len_loop

swap: sub r2, r2, #1
  ldrb r5, [r1]
  add r2, r2, r1
  ldrb r6, [r2]
  strb r6, [r1]
  strb r5, [r2]

  ldr r0, =a
  swi 0x02
  swi 0x011
