; Allocate a 3x3 matrix to store natural numbers, in column major format

  .DATA
MAT: .word 0,0,0,0,0,0,0,0,0

  .TEXT
  LDR R1, =MAT
  MOV R2, #3 ; no of rows
  MOV R3, #3 ; no of cols
  MOV R4, #1 ; sequence of natural numbers to store into matrix
  MOV R5, #0 ; row index

ROWLOOP:
  CMP R5, #3
  BEQ FIN
  MOV R6, #0 ; col index

COLLOOP:
  MLA R9, R2, R6, R5
  MOV R8, #4
  MLA R9, R8, R9, R1
  STR R4, [R9]
  ADD R4, R4, #1
  ADD R6, R6, #1
  CMP R6, #3
  BLT COLLOOP

  ADD R5, R5, #1
  B ROWLOOP

FIN: SWI 0x011
