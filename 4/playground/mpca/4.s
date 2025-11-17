  .data
A: .word 1,2,3,4,5
B: .word 1,2,3,4,5

  .text
  LDR R0, =A
  LDR R1, =B

  MOV R2, #2
  MOV R3, #3
  MOV R4, #4
  MOV R5, #5

  CMP R3, #5

  STMLTIA R0!, {R2-R5}
  STMLTEA sp!, {R2-R5}

