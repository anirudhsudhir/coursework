  ; Stores the sum or difference in r2
  
  MOV r0, #14
  MOV r1, #13

  TEQ r0,r1

  BEQ subtract

  ADD r2, r0, r1
  SWI 0x011

  subtract:
  SUB r2, r0, r1
