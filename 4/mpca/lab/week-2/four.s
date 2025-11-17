; Smallest number from a set of numbers
  .DATA
NUM: .word 10,50,41,55,30,20,11,5,100,77
ANS: .word

  .TEXT
  LDR R1, =NUM

  LDR R4, [R1], #4 ; initialising register with the first number
  MOV R2, #1 ; count

LOOP: CMP R2, #10
  BGE finish
  LDR R3, [R1], #4
  ADD R2, R2, #1

  CMP R3, R4
  BLT less_than

  B LOOP

less_than: MOV R4, R3
  B LOOP

finish: LDR R9, =ANS
  STR R4, [R9]
  SWI 0x011
