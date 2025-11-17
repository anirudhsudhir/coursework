module proj_tb;

reg [7:0] i0,i1;
wire [15:0] pdt;

carry_save_mult inst1(i0,i1,pdt);


initial
begin
i0=8'b00000000; i1=8'b00000000;
#100 i0=8'b00010111; i1=8'b00001110; //23,14
#100 i0=8'b00001011; i1=8'b00100000; //11,32
#100 i0=8'b00101101; i1=8'b00010011; //45,19
#100 i0=8'b10000001; i1=8'b00101111; //129,47
#100 i0=8'b00100010; i1=8'b01001000; //34,72
#100 i0=8'b01011011; i1=8'b01110100; //92,116
#100 i0=8'b10011101; i1=8'b00101010; //157,42
#100 i0=8'b11100111; i1=8'b00011000; //231,24
#100 i0=8'b11111111; i1=8'b11111111; //231,24
end

initial
begin
//$monitor($time," - i0 = %d, i1 = %d, g=%d, h=%d product = 0-%d 1-%d 2-%d 3-%d 4-%d 5-%d 6-%d 7-%d 8-%d 9-%d 10-%d 11-%d 12-%d 13-%d 14-%d 15-%d", i0, i1, g,h, pdt[0], pdt[1], pdt[2], pdt[3], pdt[4], pdt[5], pdt[6], pdt[7], pdt[8], pdt[10], pdt[10], pdt[11], pdt[12], pdt[13], pdt[14], pdt[15]);
$monitor($time," - i0 = %d, i1 = %d, product = %d", i0, i1, pdt);
$dumpfile("proj_tb.vcd");
$dumpvars(0, proj_tb);
end
endmodule;
