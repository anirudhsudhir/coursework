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
