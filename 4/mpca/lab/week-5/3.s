  .data
str: .asciz "hello world"
substr: .asciz "or"

  .text
  ldr r1, =str
  ldr r2, =substr

  ldrb r3, [r1], #1
  ldrb r4, [r2], #1
  mov r6, #0


str_loop: cmp r3, #0
  beq no_match

  cmp r3, r4
  moveq r5, r1

  bleq check_match

  ldrb r3, [r1], #1
  add r6, r6, #1
  b str_loop


check_match: ldrb r3, [r1], #1
  ldrb r4, [r2], #1

  cmp r4, #0 ; substring complete, match found
  beq match_found

  cmp r3, #0
  beq no_match

  cmp r3, r4
  movne r1, r5
  ldrne r2, =substr
  ldrb r4, [r2], #1
  bxne lr

  b check_match

match_found: sub r6, r6, #1
  mov r0, r6 ; r0 stores index of where substring starts
  swi 0x011

no_match: mov r0, #0
  swi 0x011
