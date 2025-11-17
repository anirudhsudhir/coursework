module not_tb;

reg i1, i2;
wire o;

not2 not2_inst(o, i1);

initial
begin
$display("Testbench for NOT gate");
i1=0;
#10 i1=0;
#10 i1=1;
#10 i1=0;
end

initial
begin
$monitor($time, " - i1 = %b, o= %b", i1, o);
end

 initial begin $dumpfile("tb.vcd"); $dumpvars(0,not_tb); end

endmodule;
