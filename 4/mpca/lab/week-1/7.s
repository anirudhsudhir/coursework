; r2 holds the computed gcd

  MOV r0, #72
  MOV r1, #48

  loop_gcd:
  CMP r0, r1
  BEQ ans
  BGT loop_div
  MOV r2, r1
  MOV r1, r0
  MOV r0, r2
  B loop_gcd

  loop_div:
  SUB r0, r0, r1
  CMP r0, r1
  BGT loop_div
  B loop_gcd

  ans:
  MOV r2, r0
