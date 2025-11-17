module and2(c,a,b);
input a,b;
output c;
assign c = a&b;
endmodule

module or2(c,a,b);
input a,b;
output c;
assign c = a|b;
endmodule

module not2(c,a);
input a;
output c;
assign c = !a;
endmodule

module nand2(c,a,b);
input a,b;
output c;
assign c = !(a&b);
endmodule

module nand3(d,a,b,c);
input a,b,c;
output d;
assign d = !(a&b&c);
endmodule

module nand4(e,a,b,c,d);
input a,b,c,d;
output e;
assign e = !(a&b&c&d);
endmodule

module nor2(c,a,b);
input a,b;
output c;
assign c = !(a|b);
endmodule

module xor2(c,a,b);
input a,b;
output c;
assign c = a^b;
endmodule

module and3(d,a,b,c);
input a,b,c;
output d;
assign d = a&b&c;
endmodule

module or3(d,a,b,c);
input a,b,c;
output d;
assign d = a|b|c;
endmodule

module or4(e,a,b,c,d);
input a,b,c,d;
output e;
assign e = a|b|c|d;
endmodule

module and4(e,a,b,c,d);
input a,b,c,d;
output e;
assign e = a&b&c&d;
endmodule

module hadder(cout,sum,a,b);
    input a,b;
    output cout,sum;
    xor2 u1(sum,a,b);
    and2 u2(cout,a,b);
endmodule

module fadder(cout, sum, a, b, cin);
    input a, b, cin;
    output cout, sum;
    wire t0, t1, t2, t3, t4;
    
    and2 u1(t1, a, b);
    and2 u2(t2, a, cin);
    and2 u3(t3, b, cin);
    or2  u4(t4, t1, t2);
    or2  u5(cout, t4, t3);
    xor2 u6(t0, a, b);
    xor2 u7(sum, cin, t0);
endmodule

module mux2(f,i0,i1,s);
    input i0,i1,s;
    output f;
    wire x1,x2,x3;
    not2 u1(x3,s);
    and2 u2(x1,x3,i0);
    and2 u3(x2,s,i1);
	or2  u4(f,x1,x2);
endmodule


module fourmux(f,i0,i1,i2,i3,s1,s2);
    input i0,i1,i2,i3,s1,s2;
	output f;
	wire x1,x2,x3,x4,x5,x6;
	not2 u1(x1,s1);
	not2 u2(x2,s2);
	and3 u3(x3,x1,x2,i0);
	and3 u4(x4,x1,s2,i1);
	and3 u5(x5,s1,x2,i2);
	and3 u6(x6,s1,s2,i3);
	or4  u7(f,x3,x4,x5,x6);
endmodule


module twoLD(D0,D1,D2,D3,E,A,B);
    input E,A,B;
	output D0,D1,D2,D3;
	wire x1,x2;
	not2 u1(x1,A);
	not2 u2(x2,B);
	and3 u3(D0,E,x1,x2);
	and3 u4(D1,E,x1,B);
	and3 u5(D2,E,A,x2);
	and3 u6(D3,E,A,B);
endmodule

module threeLD(D0,D1,D2,D3,D4,D5,D6,D7,E,A,B,C);
    input E,A,B,C;
	output D0,D1,D2,D3,D4,D5,D6,D7;
	wire x1,x2,x3;
	not2 u1(x1,A);
	not2 u2(x2,B);
	not2 u3(x3,C);
	and4 u4(D0,E,x1,x2,x3);
	and4 u5(D1,E,x1,x2,C);
	and4 u6(D2,E,x1,B,x3);
	and4 u7(D3,E,x1,B,C);
	and4 u8(D4,E,A,x2,x3);
	and4 u9(D5,E,A,x2,C);
	and4 u10(D6,E,A,B,x3);
	and4 u11(D7,E,A,B,C);
endmodule

module lowtwoLD(D0,D1,D2,D3,E,A,B);
    input E,A,B;
	output D0,D1,D2,D3;
	wire x1,x2,xE;
	not2 u0(x1,A);
	not2 u1(x2,B);
	not2 u2(xE,E);
	nand3 u3(D0,xE,x1,x2);
	nand3 u4(D1,xE,x1,B);
	nand3 u5(D2,xE,A,x2);
	nand3 u6(D3,xE,A,B);
endmodule

module lowthreeLD(D0,D1,D2,D3,D4,D5,D6,D7,E,A,B,C);
    input E, A, B, C;
    output D0,D1,D2,D3,D4,D5,D6,D7;
    wire x1,x2,x3,xE;

    not2 u0(x1, A);
    not2 u1(x2, B);
    not2 u2(x3, C);
    not2 u3(xE, E);

    nand4 u4(D0, xE, x1, x2, x3);
    nand4 u5(D1, xE, x1, x2, C);
    nand4 u6(D2, xE, x1, B, x3);
    nand4 u7(D3, xE, x1, B, C);
    nand4 u8(D4, xE, A, x2, x3);
    nand4 u9(D5, xE, A, x2, C);
    nand4 u10(D6, xE, A, B, x3);
    nand4 u11(D7, xE, A, B, C);
endmodule	

