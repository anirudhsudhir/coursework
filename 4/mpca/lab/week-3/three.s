; copy block of 128 bytes from loc A to B; rate of data transfer = 16 bytes

  .DATA
A: .WORD 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32
B: .WORD 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

  .TEXT
  LDR R1, =A
  LDR R2, =B
  MOV R3, #0

LOOP: CMP R3, #128
  BEQ FIN
  ADD R3, R3, #16
  LDMIA R1!, {R5-R8}
  STMIA R2!, {R5-R8}
  B LOOP

FIN: SWI 0x011
