; Store positive and negative numbers separately
  .DATA
A: .word 1,2,3,4,-1,5,-2,-3,6,0
POS: .word 0,0,0,0,0,0,0,0,0,0
NEG: .word 0,0,0,0,0,0,0,0,0,0

  .TEXT
  LDR R1,=A
  MOV R2,#10 ; number of elements
  LDR R8, =POS
  LDR R9, =NEG

  MOV R4, #0; count of elements read
  MOV R10, #0

LOOP: CMP R4, R2
  BGE finish
  LDR R5, [R1], #4 ; Post-indexing
  ADD R4, R4, #1
  CMP R5, #0
  BPL positive
  B negative

positive: STR R5, [R8], #4
  ADD R10, R10, R5
  B LOOP

negative: STR R5, [R9], #4
  B LOOP

finish: SWI 0x011

