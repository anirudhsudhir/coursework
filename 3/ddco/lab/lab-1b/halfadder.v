module halfadder(os,oc,i1,i2);

input i1,i2;
output os,oc;

xor(os,i1,i2);
and(oc,i1,i2);

endmodule;
