module halfadder_tb;

reg i1,i2;
wire os,oc;

halfadder inst(os,oc,i1,i2);

initial
begin
#100 i1=0;i2=0;
#100 i1=0;i2=1;
#100 i1=1;i2=0;
#100 i1=1;i2=1;
end

initial
begin
$monitor($time," - i1=%b, i2=%b, sum=%b, carry=%b",i1,i2,os,oc);
end

initial
begin
$dumpfile("./halfadder_tb.vcd");
$dumpvars(0,halfadder_tb);
end

endmodule;
