.DATA
A: .ASCIZ"dog"


.TEXT

LDR R0, =A
LDR R1, =A; 

LDRB R2, [R0]; 

ENDLOOP: 
    LDRB R3, [R1], #1; 
    CMP R3, #0 
    BNE ENDLOOP 

  SUB R1, R1, #2 ; ANIRUDH: should subtract by 2 as you want the last byte
LDRB R3, [R1];

STRB R3, [R0]; 
STRB R2, [R1]; 

MOV R0, #1
LDR R1, =A
MOV R2, #3
MOV R7, #4
SWI 0x02
SWI 0x011
