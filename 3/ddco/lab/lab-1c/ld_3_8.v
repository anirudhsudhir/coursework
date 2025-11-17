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
