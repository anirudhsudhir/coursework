; Transpose of a matrix

  .DATA

mat: .word 1,2,3
  .word 4,5,6
  .word 7,8,9

transpose: .word 0,0,0
  .word 0,0,0
  .word 0,0,0

  .TEXT
  LDR R0, =mat
  LDR R1, =transpose

  MOV R3, #0 ; row index
  MOV R6, #3 ; no of rows = no of cols

ROWLOOP: CMP R3, R6
  BEQ FIN
  MOV R4, #0 ; col index
  BL COLLOOP

  ADD R3, R3, #1
  B ROWLOOP

COLLOOP: CMP R4, R6
  BXEQ LR

  MLA R8, R3, R6, R4
  MOV R7, #4
  MLA R8, R7, R8, R0

  MLA R9, R4, R6, R3
  MLA R9, R7, R9, R1

  LDR R2, [R8]
  STR R2, [R9]

  ADD R4, R4, #1
  B COLLOOP

FIN: SWI 0x011
