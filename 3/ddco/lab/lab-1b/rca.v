module rca(sum,cout,i1,i2,cin);

input [3:0]i1,i2;
input cin;
output [3:0]sum;
output cout;

wire t1,t2,t3;

fulladder f1(sum[0],t1,i1[0],i2[0],cin);
fulladder f2(sum[1],t2,i1[1],i2[1],t1);
fulladder f3(sum[2],t3,i1[2],i2[2],t2);
fulladder f4(sum[3],cout,i1[3],i2[3],t3);

endmodule;
