  ; Sum of digits
  .DATA
NUM: .word 788

  .TEXT
  MOV R0, #0 ; R0 holds sum of digits
  LDR R1, =NUM
  LDR R2, [R1]

  MOV R3, R2
  MOV R4, #10
  MOV R5, #0

DIVIDE: CMP R3, R4
  BLT DIGIT
  SUB R3, R3, R4
  ADD R5, R5, #1
  B DIVIDE

DIGIT: ADD R0, R0, R3
  MOV R3, R5
  CMP R3, #10
  BLT FIN
  MOV R5, #0
  B DIVIDE

FIN: ADD R0, R0, R3
  SWI 0x011
