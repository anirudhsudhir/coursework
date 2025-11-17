
  ; Smallest digit in a BCD number

  .DATA
num: .word 0x78415374

  .TEXT

  LDR R1, =num
  LDR R1, [R1]
  AND R0, R1, #0xf ; initialising the register with the last digit of the number

LOOP: CMP R1, #0
  BEQ FIN
  AND R2, R1, #0xf
  MOV R1, R1, LSR #4
  CMP R2, R0
  MOVLT R0, R2
  B LOOP

FIN: SWI 0x011
