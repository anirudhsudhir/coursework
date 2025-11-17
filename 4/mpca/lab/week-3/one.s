; remainder of a number
  
  .DATA
M: .word 10
N: .word 3

  .TEXT
  LDR R1, =M
  LDR R2, =N

  LDR R3, [R1]
  LDR R4, [R2]

LOOP: CMP R3, R4
  BLT ANS
  SUB R3, R3, R4
  B LOOP

ANS: MOV R0, R3
  SWI 0x011

