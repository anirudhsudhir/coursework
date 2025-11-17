  ; Sum of elements in a given row
  ; Sum is stored in register R0

  .DATA

mat: .word 1,2,3
  .word 4,5,6
  .word 7,8,9
  .word 3,6,1

  .TEXT
  LDR R7, =mat

  MOV R1, #3 ; no of cols
  MOV R2, #0 ; col index

  MOV R3, #3 ; row index whose sum must be calculated

  BL COLLOOP
  SWI 0x011

COLLOOP: CMP R2, R1
  BXEQ LR

  MLA R4, R3, R1, R2
  MOV R5, #4
  MLA R4, R5, R4, R7

  LDR R6, [R4]
  ADD R0, R0, R6

  ADD R2, R2, #1
  B COLLOOP

