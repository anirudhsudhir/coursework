
  .data
A: .word 0,1,2,3,4,5,6,7,8

  .text

  MOV R1, #0 ;g
  LDR R3, =A

  MOV R4, #1 ;i
  MOV R7, #4

LOOP: CMP R4, #8
  BGT FIN

  MLA R6, R4, R7, R3
  LDR R5, [R6]
  ADD R1, R2, R5
  ADD R4, R4, #1
  B LOOP


FIN: SWI 0x011

