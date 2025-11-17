; find element from matrix arranged in row major if coordinates and find the memory address

  .DATA
A: .WORD 1,2,3,4,5,6,7,8,9

  .TEXT
  LDR R1, =A

  MOV R2, #3 ; num of rows
  MOV R3, #3 ; num of cols
  MOV R4, #1; x-coord
  MOV R5, #0; y-coord
  MOV R7, #4

  MLA R6, R4, R2, R5
  MUL R6, R7, R6
  ADD R1, R1, R6
  LDR R0, [R1]
