; R0 holds the final evaluated answer
 
  .data
STR: .asciz "-+73-52"
LEN: .word 7

  .text
  LDR R5, =STR
  LDR R6, =LEN
  LDR R6, [R6]
  SUB R6, R6, #1
  ADD R5, R5, R6

  LDRB R1, [R5], #-1
  MOV R2, #0

LOOP: CMP R1, #0
  BEQ FIN

  CMP R1,#43 ;+
  BLEQ PLUS

  CMP R1,#45 ;-
  BLEQ MINUS

  CMP R1, #48
  BLGE NUM

  LDRB R1, [R5], #-1
  B LOOP

PLUS: CMP R2, #2
  BLT PLUS_PUSH

  LDMFD sp!, {R3-R4}
  ADD R3, R3, R4
  STMFD sp!, {R3}
  SUB R2, R2, #1
  BX LR

PLUS_PUSH: STMFD sp!, {R1}
  BX LR

MINUS: CMP R2, #2
  BLT MINUS_PUSH

  LDMFD sp!, {R3-R4}
  SUB R3, R3, R4
  STMFD sp!, {R3}
  SUB R2, R2, #1
  BX LR

MINUS_PUSH: STMFD sp!, {R1}
  BX LR

NUM:
  SUB R1, R1, #48 ; converting ascii to int
  STMFD sp!, {R1}
  ADD R2, R2, #1

  BX LR

FIN: LDMFD sp!, {R0}
  ADD R0, R0, #48
  SWI 0x00
  SWI 0x011
