// 1-bit ALU
module alu_1(cout,o,i1,i0,op1,op0,cin);

input i1,i0,op1,op0,cin;
output cout,o;

wire xor_o,xorc,fa_sum,and_o,or_o,mux_log_o;

// Arithmetic unit
xor(xor_o, op0,i1);
fulladder fa(fa_sum,cout,xor_o,i0,cin);

// Logic unit
or(or_o,i1,i0);
and(and_o,i1,i0);
mux_2_1 m1(mux_log_o,or_o,and_o,op0);

mux_2_1 m2(o,fa_sum,mux_log_o,op1);

endmodule;


// 16-bit ALU
module alu (input wire [1:0] op, input wire [15:0] i0, i1,
    output wire [15:0] o, output wire cout);

// Declare wires here
wire [14:0] c;

// Instantiate modules here

alu_1 a1(c[0],o[0],i0[0],i1[0],op[1],op[0],op[0]);
alu_1 a2(c[1],o[1],i0[1],i1[1],op[1],op[0],c[0]);
alu_1 a3(c[2],o[2],i0[2],i1[2],op[1],op[0],c[1]);
alu_1 a4(c[3],o[3],i0[3],i1[3],op[1],op[0],c[2]);
alu_1 a5(c[4],o[4],i0[4],i1[4],op[1],op[0],c[3]);
alu_1 a6(c[5],o[5],i0[5],i1[5],op[1],op[0],c[4]);
alu_1 a7(c[6],o[6],i0[6],i1[6],op[1],op[0],c[5]);
alu_1 a8(c[7],o[7],i0[7],i1[7],op[1],op[0],c[6]);
alu_1 a9(c[8],o[8],i0[8],i1[8],op[1],op[0],c[7]);
alu_1 a10(c[9],o[9],i0[9],i1[9],op[1],op[0],c[8]);
alu_1 a11(c[10],o[10],i0[10],i1[10],op[1],op[0],c[9]);
alu_1 a12(c[11],o[11],i0[11],i1[11],op[1],op[0],c[10]);
alu_1 a13(c[12],o[12],i0[12],i1[12],op[1],op[0],c[11]);
alu_1 a14(c[13],o[13],i0[13],i1[13],op[1],op[0],c[12]);
alu_1 a15(c[14],o[14],i0[14],i1[14],op[1],op[0],c[13]);
alu_1 a16(cout,o[15],i0[15],i1[15],op[1],op[0],c[14]);

endmodule
