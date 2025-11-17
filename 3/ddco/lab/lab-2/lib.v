// NOT gate
module not2(o, i1);

output o;
input i1;

assign o= ~i1;

endmodule;


// OR gate
module or2(o, i1, i2);

output o;
input i1, i2;

assign o=i1|i2;

endmodule;


// AND gate
module and2(o, i1, i2);

output o;
input i1, i2;

assign o=i1&i2;

endmodule;


// NAND gate
module nand2(o, i1, i2);

output o;
input i1, i2;

assign o=~(i1&i2);

endmodule;

// NOR gate
module nor2(o, i1, i2);

output o;
input i1, i2;

assign o=~(i1|i2);

endmodule;

// XOR gate
module xor2(o, i1, i2);

output o;
input i1, i2;

assign o=i1^i2;

endmodule;

// XNOR gate
module xnor2(o, i1, i2);

output o;
input i1, i2;

assign o=~(i1^i2);

endmodule;


// Half adder
module halfadder(os,oc,i1,i2);

input i1,i2;
output os,oc;

xor(os,i1,i2);
and(oc,i1,i2);

endmodule;


// Full adder
module fulladder(os,oc,i1,i2,i3);

output os, oc;
input i1, i2, i3;

wire t1,t2,t3;

halfadder ha1(t1,t2,i1,i2);
halfadder ha2(os,t3,t1,i3);
or(oc,t2,t3);

endmodule;


// 4-bit ripple carry adder
module rca(sum,cout,i1,i2,cin);

input [3:0]i1,i2;
input cin;
output [3:0]sum;
output cout;

wire t1,t2,t3;

fulladder f1(sum[0],t1,i1[0],i2[0],cin);
fulladder f2(sum[1],t2,i1[1],i2[1],t1);
fulladder f3(sum[2],t3,i1[2],i2[2],t2);
fulladder f4(sum[3],cout,i1[3],i2[3],t3);

endmodule;


// 2-4 Line decoder
module ld_2_4(i0,i1,i2,i3,a,b);

output i0,i1,i2,i3;
input a,b;

wire nota,notb;

not(nota,a);
not(notb,b);

and(i0,nota,notb);
and(i1,nota,b);
and(i2,a,notb);
and(i3,a,b);

endmodule;


// 3-8 Line decoder
module ld_3_8(i0,i1,i2,i3,i4,i5,i6,i7,s2,s1,s0);

output i0,i1,i2,i3,i4,i5,i6,i7;
input s2,s1,s0;

wire nots2,nots1,nots0;

not(nots2,s2);
not(nots1,s1);
not(nots0,s0);

and(i0,nots2,nots1,nots0);
and(i1,nots2,nots1,s0);
and(i2,nots2,s1,nots0);
and(i3,nots2,s1,s0);
and(i4,s2,nots1,nots0);
and(i5,s2,nots1,s0);
and(i6,s2,s1,nots0);
and(i7,s2,s1,s0);

endmodule;


// 2-1 Multiplexer
module mux_2_1(o,i0,i1,s0);

output o;
input i0,i1,s0;

wire nots0,and1,and2;

not(nots0,s0);

and(and1,i0,nots0);
and(and2,i1,s0);

or(o,and1,and2);

endmodule;


// 4-1 Multiplexer
module mux_4_1(o,i0,i1,i2,i3,s0,s1);

output o;
input i0,i1,i2,i3,s0,s1;

wire nots0,nots1,and1,and2,and3,and4;

not(nots0,s0);
not(nots1,s1);

and(and1,i0,nots1,nots0);
and(and2,i1,nots1,s0);
and(and3,i2,s1,nots0);
and(and4,i3,s1,s0);

or(o,and1,and2,and3,and4);

endmodule;


// 2-4 Line Decoder with enable input
module ld_2_4_enable(i0,i1,i2,i3,a,b,e);

output i0,i1,i2,i3;
input a,b,e;

wire nota,notb;

not(nota,a);
not(notb,b);

and(i0,nota,notb,e);
and(i1,nota,b,e);
and(i2,a,notb,e);
and(i3,a,b,e);

endmodule;


// 3-8 Line Decoder with enable input
module ld_3_8_enable(i0,i1,i2,i3,i4,i5,i6,i7,s2,s1,s0,e);

output i0,i1,i2,i3,i4,i5,i6,i7;
input s2,s1,s0,e;

wire nots2,nots1,nots0;

not(nots2,s2);
not(nots1,s1);
not(nots0,s0);

and(i0,nots2,nots1,nots0,e);
and(i1,nots2,nots1,s0,e);
and(i2,nots2,s1,nots0,e);
and(i3,nots2,s1,s0,e);
and(i4,s2,nots1,nots0,e);
and(i5,s2,nots1,s0,e);
and(i6,s2,s1,nots0,e);
and(i7,s2,s1,s0,e);

endmodule;

