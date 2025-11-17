// Write code for modules you need here

module pc (input wire clk, reset, inc, add, sub, input wire [15:0] offset, output wire [15:0] pc);

// Declare wires here
wire [15:0] fa_o, fa_c, fa_in, dfrl_o, and_o;
wire ias_in;
wire not_inc;
wire ias_or;

// Instantiate modules here

not n1(not_inc, inc);
or3 or3_1(inc, add, sub, ias_or);

or2 or_1(inc, offset[0],and_o[0]);
and2 and_2(not_inc, offset[1], and_o[1]);
and2 and_3(not_inc, offset[2], and_o[2]);
and2 and_4(not_inc, offset[3], and_o[3]);
and2 and_5(not_inc, offset[4], and_o[4]);
and2 and_6(not_inc, offset[5], and_o[5]);
and2 and_7(not_inc, offset[6], and_o[6]);
and2 and_8(not_inc, offset[7], and_o[7]);
and2 and_9(not_inc, offset[8], and_o[8]);
and2 and_10(not_inc, offset[9], and_o[9]);
and2 and_11(not_inc, offset[10], and_o[10]);
and2 and_12(not_inc, offset[11], and_o[11]);
and2 and_13(not_inc, offset[12], and_o[12]);
and2 and_14(not_inc, offset[13], and_o[13]);
and2 and_15(not_inc, offset[14], and_o[14]);
and2 and_16(not_inc, offset[15], and_o[15]);

xor2 xor_1(and_o[0], sub, fa_in[0]);
xor2 xor_2(and_o[1], sub, fa_in[1]);
xor2 xor_3(and_o[2], sub, fa_in[2]);
xor2 xor_4(and_o[3], sub, fa_in[3]);
xor2 xor_5(and_o[4], sub, fa_in[4]);
xor2 xor_6(and_o[5], sub, fa_in[5]);
xor2 xor_7(and_o[6], sub, fa_in[6]);
xor2 xor_8(and_o[7], sub, fa_in[7]);
xor2 xor_9(and_o[8], sub, fa_in[8]);
xor2 xor_10(and_o[9], sub, fa_in[9]);
xor2 xor_11(and_o[10], sub, fa_in[10]);
xor2 xor_12(and_o[11], sub, fa_in[11]);
xor2 xor_13(and_o[12], sub, fa_in[12]);
xor2 xor_14(and_o[13], sub, fa_in[13]);
xor2 xor_15(and_o[14], sub, fa_in[14]);
xor2 xor_16(and_o[15], sub, fa_in[15]);

fulladder fa1(fa_o[0], fa_c[0], fa_in[0], dfrl_o[0], sub);
fulladder fa2(fa_o[1], fa_c[1], fa_in[1], dfrl_o[1], fa_c[0]);
fulladder fa3(fa_o[2], fa_c[2], fa_in[2], dfrl_o[2], fa_c[1]);
fulladder fa4(fa_o[3], fa_c[3], fa_in[3], dfrl_o[3], fa_c[2]);
fulladder fa5(fa_o[4], fa_c[4], fa_in[4], dfrl_o[4], fa_c[3]);
fulladder fa6(fa_o[5], fa_c[5], fa_in[5], dfrl_o[5], fa_c[4]);
fulladder fa7(fa_o[6], fa_c[6], fa_in[6], dfrl_o[6], fa_c[5]);
fulladder fa8(fa_o[7], fa_c[7], fa_in[7], dfrl_o[7], fa_c[6]);
fulladder fa9(fa_o[8], fa_c[8], fa_in[8], dfrl_o[8], fa_c[7]);
fulladder fa10(fa_o[9], fa_c[9], fa_in[9], dfrl_o[9], fa_c[8]);
fulladder fa11(fa_o[10], fa_c[10], fa_in[10], dfrl_o[10], fa_c[9]);
fulladder fa12(fa_o[11], fa_c[11], fa_in[11], dfrl_o[11], fa_c[10]);
fulladder fa13(fa_o[12], fa_c[12], fa_in[12], dfrl_o[12], fa_c[11]);
fulladder fa14(fa_o[13], fa_c[13], fa_in[13], dfrl_o[13], fa_c[12]);
fulladder fa15(fa_o[14], fa_c[14], fa_in[14], dfrl_o[14], fa_c[13]);
fulladder fa16(fa_o[15], fa_c[15], fa_in[15], dfrl_o[15], fa_c[14]);

dfrl dfrl_1(clk, reset, ias_or, fa_o[0], dfrl_o[0]);
dfrl dfrl_2(clk, reset, ias_or, fa_o[1], dfrl_o[1]);
dfrl dfrl_3(clk, reset, ias_or, fa_o[2], dfrl_o[2]);
dfrl dfrl_4(clk, reset, ias_or, fa_o[3], dfrl_o[3]);
dfrl dfrl_5(clk, reset, ias_or, fa_o[4], dfrl_o[4]);
dfrl dfrl_6(clk, reset, ias_or, fa_o[5], dfrl_o[5]);
dfrl dfrl_7(clk, reset, ias_or, fa_o[6], dfrl_o[6]);
dfrl dfrl_8(clk, reset, ias_or, fa_o[7], dfrl_o[7]);
dfrl dfrl_9(clk, reset, ias_or, fa_o[8], dfrl_o[8]);
dfrl dfrl_10(clk, reset, ias_or, fa_o[9], dfrl_o[9]);
dfrl dfrl_11(clk, reset, ias_or, fa_o[10], dfrl_o[10]);
dfrl dfrl_12(clk, reset, ias_or, fa_o[11], dfrl_o[11]);
dfrl dfrl_13(clk, reset, ias_or, fa_o[12], dfrl_o[12]);
dfrl dfrl_14(clk, reset, ias_or, fa_o[13], dfrl_o[13]);
dfrl dfrl_15(clk, reset, ias_or, fa_o[14], dfrl_o[14]);
dfrl dfrl_16(clk, reset, ias_or, fa_o[15], dfrl_o[15]);

assign pc = dfrl_o;
endmodule

// Full adder
module fulladder(os,oc,i1,i2,i3);

output os, oc;
input i1, i2, i3;

wire t1,t2,t3;

halfadder ha1(t1,t2,i1,i2);
halfadder ha2(os,t3,t1,i3);
or(oc,t2,t3);

endmodule;

// Half adder
module halfadder(os,oc,i1,i2);

input i1,i2;
output os,oc;

xor(os,i1,i2);
and(oc,i1,i2);

endmodule;

