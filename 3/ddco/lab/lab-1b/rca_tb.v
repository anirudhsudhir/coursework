module rca_tb;

wire [3:0]sum;
wire carry;

reg [3:0]i1,i2;
reg cin;

rca rca_inst(sum,carry,i1,i2,cin);

initial
begin
i1=4'b1010;i2=4'b1100;cin=0;
#10 i1=4'b0011;i2=4'b1001;
#10 i1=4'b0000;i2=4'b0100;
#10 i1=4'b0100;i2=4'b1111;
#10 i1=4'b1111;i2=4'b1111;
end

initial
begin
$monitor($time, " - i1=%d,i2=%d,sum=%d,carry=%d",i1,i2,sum,carry);
end

initial
begin
$dumpfile("rca_tb.vcd");
$dumpvars(0, rca_tb);
end
endmodule;
