module fulladder(os,oc,i1,i2,i3);

output os, oc;
input i1, i2, i3;

wire t1,t2,t3;

halfadder ha1(t1,t2,i1,i2);
halfadder ha2(os,t3,t1,i3);
or(oc,t2,t3);

endmodule;
