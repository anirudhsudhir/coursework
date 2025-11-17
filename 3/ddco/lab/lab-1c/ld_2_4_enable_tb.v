module ld_2_4_enable_tb;

reg a,b,enable;
wire i0,i1,i2,i3;

ld_2_4_enable ld(i0,i1,i2,i3,a,b,enable);

initial
begin
a=0;b=0;enable=1;
#100 a=0;b=1;
#100 a=1;b=0;
#100 a=1;b=1;
end

initial
begin
$monitor($time, " - i0=%b,i1=%b,i2=%b,i3=%b,a=%b,b=%b,enable=%b",i0,i1,i2,i3,a,b,enable);
$dumpfile("./ld_2_4_enable_tb.vcd");
$dumpvars(0,ld_2_4_enable_tb);
end

endmodule;
