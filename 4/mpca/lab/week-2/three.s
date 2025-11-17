; add only negative numbers

  .DATA
N: .word 1,2,3,4,-1,5,-2,-3,6,0
NSUM: .word 0

  .TEXT
  LDR R1, =N
  MOV R3, #0

LOOP: CMP R3, #10
  BGE finish
  LDR R4, [R1], #4
  ADD R3, R3, #1
  CMP R4, #0
  BMI negative
  B LOOP

negative: ADD R5, R5, R4
  B LOOP

finish: LDR R6, =NSUM
  STR R5, [R6]
  SWI 0x011

