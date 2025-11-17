  ; NOT of 5 = xFFFFFFFA
  ; Stored in r2
  MOV r0,#5
  MOV r1,#0xFFFFFFFF

  EOR r2, r1, r0 ; r2 - xFFFFFFFA
