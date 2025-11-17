.data
a: .asciz "hello world!"

  .text
  ldr r1, =a
  mov r0, #0

  add r3, r1, r0
  ldrb r4, [r3]

len_loop: cmp r4, #0
  beq fin
  add r0, r0, #1

  add r3, r1, r0
  ldrb r4, [r3]

  b len_loop


fin: swi 0x011
