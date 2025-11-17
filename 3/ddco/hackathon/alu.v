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
