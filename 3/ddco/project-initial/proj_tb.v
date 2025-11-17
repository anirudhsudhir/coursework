module proj_tb;

reg [7:0]i0,i1;
reg cin;
wire [8:0]sum;

carry_save inst1(i0,i1,cin,sum);

initial
begin
i0=8'b00000000; i1=8'b00000000; cin=0;
#100 i0=8'b00010111; i1=8'b00001110; cin=0; //23,14
#100 i0=8'b00001011; i1=8'b00100000; cin=1; //11,32
#100 i0=8'b00101101; i1=8'b00010011; cin=0; //45,19
#100 i0=8'b10000001; i1=8'b00101111; cin=0; //129,47
#100 i0=8'b00100010; i1=8'b01001000; cin=1; //34,72
#100 i0=8'b01011011; i1=8'b01110100; cin=0; //92,116
#100 i0=8'b10011101; i1=8'b00101010; cin=0; //157,42
#100 i0=8'b11100111; i1=8'b00011000; cin=1; //231,24
end

initial
begin
$monitor($time," - i0 = %d, i1 = %d, cin = %b, sum = %d", i0, i1, cin, sum);
$dumpfile("proj_tb.vcd");
$dumpvars(0, proj_tb);
end
endmodule;
