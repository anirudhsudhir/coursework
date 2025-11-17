  ; Factorial of 5 = 120 = 78 in hexadecimal
  ; Stored in r1
  
  MOV r0, #5
  MOV r1, #1

loop:
  MUL r1, r0, r1
  SUB r0, r0, #1
  CMP r0, #1
  BGT loop
