module mux_2_1(o,i0,i1,s0);

output o;
input i0,i1,s0;

wire nots0,and1,and2;

not(nots0,s0);

and(and1,i0,nots0);
and(and2,i1,s0);

or(o,and1,and2);

endmodule;
