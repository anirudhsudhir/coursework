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
