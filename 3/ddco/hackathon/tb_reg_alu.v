`timescale 1 ns / 100 ps
`define TESTVECS 8

module tb;
  reg clk, reset, wr, sel;
  reg [1:0] op;
  reg [2:0] rd_addr_a, rd_addr_b, wr_addr; reg [7:0] d_in;
  wire [7:0] d_out_a, d_out_b;
  reg [20:0] test_vecs [0:(`TESTVECS-1)];
  integer i;
  initial begin $dumpfile("tb_reg_alu.vcd"); $dumpvars(0,tb); end
  initial begin reset = 1'b1; #12.5 reset = 1'b0; end
  initial clk = 1'b0; always #5 clk =~ clk;
  initial begin
    test_vecs[0][20] = 1'b0; test_vecs[0][19] = 1'b1; test_vecs[0][18:17] = 2'b00;
    test_vecs[0][16:14] = 3'ox; test_vecs[0][13:11] = 3'ox;
    test_vecs[0][10:8] = 3'h3; test_vecs[0][7:0] = 8'o6;

    test_vecs[1][20] = 1'b0; test_vecs[1][19] = 1'b1; test_vecs[1][18:17] = 2'b01;
    test_vecs[1][16:14] = 3'o3; test_vecs[1][13:11] = 3'o7;
    test_vecs[1][10:8] = 3'o7; test_vecs[1][7:0] = 8'o4;

    test_vecs[2][20] = 1'b0; test_vecs[2][19] = 1'b1; test_vecs[2][18:17] = 2'b10;
    test_vecs[2][16:14] = 3'o3; test_vecs[2][13:11] = 3'o7;
    test_vecs[2][10:8] = 3'o5; test_vecs[2][7:0] = 8'o2;

    test_vecs[3][20] = 1'b0; test_vecs[3][19] = 1'b1; test_vecs[3][18:17] = 2'b11;
    test_vecs[3][16:14] = 3'o1; test_vecs[3][13:11] = 3'o5;
    test_vecs[3][10:8] = 3'o1; test_vecs[3][7:0] = 8'o5;

    test_vecs[4][20] = 1'b1; test_vecs[4][19] = 1'b1; test_vecs[4][18:17] = 2'b00;
    test_vecs[4][16:14] = 3'o1; test_vecs[4][13:11] = 3'o5;
    test_vecs[4][10:8] = 3'h3; test_vecs[4][7:0] = 8'o6;

    test_vecs[5][20] = 1'b1; test_vecs[5][19] = 1'b1; test_vecs[5][18:17] = 2'b01;
    test_vecs[5][16:14] = 3'o5; test_vecs[5][13:11] = 3'o7;
    test_vecs[5][10:8] = 3'o7; test_vecs[5][7:0] = 8'o4;

    test_vecs[6][20] = 1'b1; test_vecs[6][19] = 1'b1; test_vecs[6][18:17] = 2'b10;
    test_vecs[6][16:14] = 3'o3; test_vecs[6][13:11] = 3'o7;
    test_vecs[6][10:8] = 3'o5; test_vecs[6][7:0] = 8'o2;

    test_vecs[7][20] = 1'b1; test_vecs[7][19] = 1'b1; test_vecs[7][18:17] = 2'b11;
    test_vecs[7][16:14] = 3'o1; test_vecs[7][13:11] = 3'o5;
    test_vecs[7][10:8] = 3'o1; test_vecs[7][7:0] = 8'o5;

  end
  initial begin {sel, wr, op, rd_addr_a, rd_addr_b, wr_addr, d_in} = 0;
end
  reg_alu reg_alu_0 (clk, reset, sel, wr, op, rd_addr_a, rd_addr_b, wr_addr, d_in,
  d_out_a, d_out_b, cout);
  initial begin
    #6 for(i=0;i<`TESTVECS;i=i+1)
      begin #10 {sel, wr, op, rd_addr_a, rd_addr_b, wr_addr, d_in}=test_vecs[i]; end
    #100 $finish;
  end

  initial begin 
    $monitor($time, " - sel = %b, op = %b, d_out_a=%b, d_out_b =%b, rd_addr_a=%b, rd_addr_b=%b, wr_addr=%b, d_in=%b\n",
      sel, op, d_out_a, d_out_b, rd_addr_a, rd_addr_b, wr_addr, d_in);
  end

endmodule
