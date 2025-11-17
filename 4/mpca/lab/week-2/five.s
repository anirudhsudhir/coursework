; fibonacci
  .DATA
VALS: .word 0,0,0,0,0,0,0,0,0,0

  .TEXT
  MOV R1, #0
  MOV R2, #1

  LDR R6, =VALS
  STR R1, [R6], #4
  STR R2, [R6], #4

  MOV R5, #2 ; count

LOOP: CMP R5, #10
  BGE finish

  ADD R5, R5, #1

  ADD R3, R1, R2
  STR R3, [R6], #4
  MOV R1, R2
  MOV R2, R3

  B LOOP

finish: SWI 0x011
