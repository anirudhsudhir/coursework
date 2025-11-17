module mux_2_1_tb;

wire o;
reg i0,i1,s0;

mux_2_1 mux(o,i0,i1,s0);

initial
begin
i0=0;i1=1;s0=0;
#100 s0=0;
#100 s0=1;
#100 s0=1;
end

initial
begin
$monitor($time,": o=%b,i0=%b,i1=%b,s0=%b",o,i0,i1,s0);
$dumpfile("./mux_2_1_tb.vcd");
$dumpvars(0,mux_2_1_tb);
end

endmodule;
