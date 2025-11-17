  ; convolution using MUL

  .DATA
A: .word 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16
KERNEL: .word 2,4,6,8

  .TEXT
  LDR R1, =A
  LDR R2, =KERNEL

  MOV R3, #0 ; row loop register
  MOV R4, #0 ; column loop register

ROWLOOP: CMP R3, #3 ; 4 rows - multiply with kernel with 2 rows
  BEQ FIN
  MOV R4, #0

COLLOOP: CMP R4, #3 ; 4 cols - multiply with kernel with 2 cols
  BEQ ROWLOOPRET
  B MULOOP1

MULOOP1: MOV R5, #0 ; kernel row index
  MOV R6, #0; kernel col index

MULOOP2: CMP R5, #2
  BEQ COLLOOPRET
  MOV R6, #0
  B MULOOP3

MULOOP3: CMP R6, #2
  BEQ MULOOP2RET

  ADD R7, R3, R5
  MOV R8, #4
  MUL R7, R8, R7 
  ADD R7, R4, R7
  ADD R7, R6, R7
  MUL R0, R8, R7
  ADD R0, R0, R1 ; getting index of actual matrix element to be multiplied
  LDR R8, [R0]
  MOV R7, #2 ; 2 rows in kernel
  MUL R7, R5, R7
  ADD R7, R7, R6
  MOV R9, #4
  MUL R7, R9, R7
  ADD R7, R7, R2 ; getting index of actual matrix element to be multiplied
  LDR R9, [R7]
  MUL R9, R8, R9
  STR R9, [R0]

  ADD R6, R6, #1
  B MULOOP3

MULOOP2RET: ADD R5, R5, #1
  B MULOOP2

COLLOOPRET: ADD R4, R4, #1
  B COLLOOP

ROWLOOPRET: ADD R3, R3, #1
  B ROWLOOP

FIN: SWI 0x011
