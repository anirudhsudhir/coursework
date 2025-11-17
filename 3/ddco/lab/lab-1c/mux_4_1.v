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
