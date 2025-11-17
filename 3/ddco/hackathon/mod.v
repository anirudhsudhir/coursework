module fa (input wire i0, i1, cin, output wire sum, cout);
   wire t0, t1, t2;
   xor3 _i0 (i0, i1, cin, sum);
   and2 _i1 (i0, i1, t0);
   and2 _i2 (i1, cin, t1);
   and2 _i3 (cin, i0, t2);
   or3 _i4 (t0, t1, t2, cout);
endmodule

module alu_slice (input wire [1:0] op, input wire i0, i1, cin, output wire o, cout);
   wire t_and, t_or, t_andor, t_nand, t_nor, t_nandnor;
   and2 _i1 (i0, i1, t_and);
   or2 _i2 (i0, i1, t_or);
   nand2 _i3 (i0, i1, t_nand);
   nor2 _i4 (i0, i1, t_nor);
   mux2 _i5 (t_and, t_or, op[0], t_andor);
   mux2 _i6 (t_nand, t_nor, op[0], t_nandnor);
   mux2 _i7 (t_andor, t_nandnor, op[1], o);
endmodule

module alu (input wire [1:0] op, input wire [7:0] i0, i1,
    output wire [7:0] o, output wire cout);
   wire [6:0] c;
   alu_slice _i0 (op, i0[0], i1[0], op[0] , o[0], c[0]);
   alu_slice _i1 (op, i0[1], i1[1], c[0], o[1], c[1]);
   alu_slice _i2 (op, i0[2], i1[2], c[1], o[2], c[2]);
   alu_slice _i3 (op, i0[3], i1[3], c[2], o[3], c[3]);
   alu_slice _i4 (op, i0[4], i1[4], c[3], o[4], c[4]);
   alu_slice _i5 (op, i0[5], i1[5], c[4], o[5], c[5]);
   alu_slice _i6 (op, i0[6], i1[6], c[5], o[6], c[6]);
   alu_slice _i7 (op, i0[7], i1[7], c[6], o[7], cout);
endmodule

module dfrl_8 (input wire clk, reset, load, input wire [7:0] in, output wire [7:0] out);
dfrl _f0(clk,reset,load,in[0],out[0]);
dfrl _f1(clk,reset,load,in[1],out[1]);
dfrl _f2(clk,reset,load,in[2],out[2]);
dfrl _f3(clk,reset,load,in[3],out[3]);
dfrl _f4(clk,reset,load,in[4],out[4]);
dfrl _f5(clk,reset,load,in[5],out[5]);
dfrl _f6(clk,reset,load,in[6],out[6]);
dfrl _f7(clk,reset,load,in[7],out[7]);
endmodule

module mux8_8 (input wire [0:7] i0, i1, i2, i3, i4, i5, i6, i7, input wire [0:2] j, output wire [0:7] o);
  mux8 mux8_0({i0[0], i1[0], i2[0], i3[0], i4[0], i5[0], i6[0], i7[0]}, j[0], j[1], j[2], o[0]);
  mux8 mux8_1({i0[1], i1[1], i2[1], i3[1], i4[1], i5[1], i6[1], i7[1]}, j[0], j[1], j[2], o[1]);
  mux8 mux8_2({i0[2], i1[2], i2[2], i3[2], i4[2], i5[2], i6[2], i7[2]}, j[0], j[1], j[2], o[2]);
  mux8 mux8_3({i0[3], i1[3], i2[3], i3[3], i4[3], i5[3], i6[3], i7[3]}, j[0], j[1], j[2], o[3]);
  mux8 mux8_4({i0[4], i1[4], i2[4], i3[4], i4[4], i5[4], i6[4], i7[4]}, j[0], j[1], j[2], o[4]);
  mux8 mux8_5({i0[5], i1[5], i2[5], i3[5], i4[5], i5[5], i6[5], i7[5]}, j[0], j[1], j[2], o[5]);
  mux8 mux8_6({i0[6], i1[6], i2[6], i3[6], i4[6], i5[6], i6[6], i7[6]}, j[0], j[1], j[2], o[6]);
  mux8 mux8_7({i0[7], i1[7], i2[7], i3[7], i4[7], i5[7], i6[7], i7[7]}, j[0], j[1], j[2], o[7]);
endmodule

module reg_file (input wire clk, reset, wr, input wire [2:0] rd_addr_a, rd_addr_b, wr_addr, input wire [7:0] d_in, output wire [7:0] d_out_a, d_out_b);

wire [0:7] load;
wire [7:0] r0_1,r1_2,r2_3,r3_4,r4_5,r5_6,r6_7,r_out;

dfrl_8 r0(clk, reset, load[0], d_in, r0_1);
dfrl_8 r1(clk, reset, load[1], d_in, r1_2);
dfrl_8 r2(clk, reset, load[2], d_in, r2_3);
dfrl_8 r3(clk, reset, load[3], d_in, r3_4);
dfrl_8 r4(clk, reset, load[4], d_in, r4_5);
dfrl_8 r5(clk, reset, load[5], d_in, r5_6);
dfrl_8 r6(clk, reset, load[6], d_in, r6_7);
dfrl_8 r7(clk, reset, load[7], d_in, r_out);


mux8_8 read_bus_a(r0_1,r1_2,r2_3,r3_4,r4_5,r5_6,r6_7,r_out, rd_addr_a, d_out_a);
mux8_8 read_bus_b(r0_1,r1_2,r2_3,r3_4,r4_5,r5_6,r6_7,r_out, rd_addr_b, d_out_b);

demux8 d_mux(wr, wr_addr[0],wr_addr[1],wr_addr[2],load);

endmodule


module reg_alu (input wire clk, reset, sel, wr, input wire [1:0] op, input wire [2:0] rd_addr_a,
		rd_addr_b, wr_addr, input wire [7:0] d_in, output wire [7:0] d_out_a, d_out_b, output wire cout);

wire [7:0] out_mux_wire, reg_in_wire, douta, doutb;
wire cout_val;

mux2 alu_mux0(d_in[0],out_mux_wire[0],sel,reg_in_wire[0]);
mux2 alu_mux1(d_in[1],out_mux_wire[1],sel,reg_in_wire[1]);
mux2 alu_mux2(d_in[2],out_mux_wire[2],sel,reg_in_wire[2]);
mux2 alu_mux3(d_in[3],out_mux_wire[3],sel,reg_in_wire[3]);
mux2 alu_mux4(d_in[4],out_mux_wire[4],sel,reg_in_wire[4]);
mux2 alu_mux5(d_in[5],out_mux_wire[5],sel,reg_in_wire[5]);
mux2 alu_mux6(d_in[6],out_mux_wire[6],sel,reg_in_wire[6]);
mux2 alu_mux7(d_in[7],out_mux_wire[7],sel,reg_in_wire[7]);

reg_file reg1(clk, reset, wr, rd_addr_a, rd_addr_b, wr_addr, reg_in_wire , douta, doutb);
alu alu1(op, douta, doutb, out_mux_wire, cout_val);

assign d_out_a = douta;
assign d_out_b = doutb;

dfr dfr1(clk, reset, cout_val, cout);

endmodule

