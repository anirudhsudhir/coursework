
module tb_alu_1;

reg i1,i2,i3;
wire os,oc;

alu_1 inst(oc,os,i1,i2,1'b0,1'b0,i3);

initial
begin
  i1=0;i2=0;i3=0;
  #100 i1=0;i2=0;i3=1;
  #100 i1=0;i2=1;i3=0;
  #100 i1=0;i2=1;i3=1;
  #100 i1=1;i2=0;i3=0;
  #100 i1=1;i2=0;i3=1;
  #100 i1=1;i2=1;i3=0;
  #100 i1=1;i2=1;i3=1;
end

initial
begin
  $monitor("i1 = %b, i2 = %b, i3=%b, sum = %b, carry = %b", i1, i2, i3, os, oc);
end

initial
begin
$dumpfile("tb_alu_1.vcd");
$dumpvars(0,tb_alu_1);
end

endmodule;
