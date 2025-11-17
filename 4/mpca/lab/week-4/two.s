  ; Sum of 2 BCD numbers in a function using stack parameter passing
  ; Sum is stored in R0

  .TEXT
  MOV R1, #0x78
  STMEA sp!, {R1}
  MOV R1, #0x57
  STMEA sp!, {R1}

  BL SUM_BCD
  LDMEA sp!, {R0}
  SWI 0x011

SUM_BCD: LDMEA sp!, {R2}
  LDMEA sp!, {R6}
  MOV R7, #0
LOOP: CMP R2, #0
  BEQ FIN
  CMP R6, #0
  BEQ FIN

  MOV R5, #0x0f
  AND R3, R2, R5
  ADD R4, R4, R3, LSL R7

  AND R3, R6, R5
  ADD R4, R4, R3, LSL R7

  ADD R7, R7, #4

  MOV R2, R2, LSR #4
  MOV R6, R6, LSR #4

  B LOOP

FIN: STMEA sp!, {R4}
  BX lr
