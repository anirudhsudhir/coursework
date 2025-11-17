module mux_4_1_tb;

wire o;
reg i0,i1,i2,i3,s0,s1;

mux_4_1 mux(o,i0,i1,i2,i3,s0,s1);

initial
begin
i0=1;i1=1;i2=1;i3=1;
s0=0;s1=0;
#100 s1=0;s0=1;
#100 s1=1;s0=0;
#100 s1=1;s0=1;
end

initial
begin
$display($time,": i0=%b,i1=%b,i2=%b,i3=%b",i0,i1,i2,i3);
$monitor($time,": o=%b,s1=%b,s0=%b",o,s1,s0);
$dumpfile("./mux_4_1_tb.vcd");
$dumpvars(0,mux_4_1_tb);
end

endmodule;
