  ; 1's complement of 5 - xFFFFFFFA
  ; 2's complement of 5 - xFFFFFFFB
  ; 2's complement stored in r1
  
  MOV r0,#5 ; 0x00000005
  MOV r1,#0xFFFFFFFF

  RSB r0,r0,r1 ; r0 = xFFFFFFFA

  RSB r1, r0, #0 ; r1 = -r0
  RSB r1, r1, #1 ; r1 = 1-(-r0) = 1+r0
