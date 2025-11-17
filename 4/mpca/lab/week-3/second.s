; search for element in array

  .DATA
ARR: .HWORD 1,2,3,4,5,6,7,8,9,10

  .TEXT
  LDR R1, =ARR
  MOV R4, #1 ; array count
  MOV R2, #7 ; search element
  MOV R5, #0 ; flag - set to 1 if found

ILOOP: CMP R4, #10
  BGT END
  ADD R4, R4, #1
  LDRH R3, [R1]  ; array element
  ADD R1, R1, #2
  BLT OLOOP

OLOOP: CMP R3, R2
  BEQ FOUND
  B ILOOP

FOUND: MOV R5, #1
  SWI 0x011

END: SWI 0x011
