
  .data
A: .word 8,3,7,5,2,6,9,4,1,0
LEN: .word 10

  .text

  LDR R0, =A
  LDR R1, =LEN
  LDR R1, [R1]
  SUB R1, R1, #1

  MOV R2, #0 ;i

OLOOP: CMP R2, R1
  BGE FIN

  ADD R3, R2, #1; j = i+1
  LDR R4, [R0, R2, LSL #2] ; ith array element
  BL ILOOP

  ADD R2, R2, #1
  B OLOOP

ILOOP: CMP R3, R1
  BXGT LR

  LDR R5, [R0, R3, LSL #2] ; jth array element
  CMP R5, R4
  STRLT R5, [R0, R2, LSL #2]
  STRLT R4, [R0, R3, LSL #2]
  MOVLT R4, R5

  ADD R3, R3, #1
  B ILOOP

FIN: SWI 0x011
