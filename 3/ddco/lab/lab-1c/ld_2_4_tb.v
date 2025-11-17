module ld_2_4_tb;

reg a,b;
wire i0,i1,i2,i3;

ld_2_4 ld(i0,i1,i2,i3,a,b);

initial
begin
a=0;b=0;
#100 a=0;b=1;
#100 a=1;b=0;
#100 a=1;b=1;
end

initial
begin
$monitor($time, " - i0=%b,i1=%b,i2=%b,i3=%b,a=%b,b=%b",i0,i1,i2,i3,a,b);
$dumpfile("./ld_2_4_tb.vcd");
$dumpvars(0,ld_2_4_tb);
end

endmodule;
