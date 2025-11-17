module carry_save(input [7:0] a, input [7:0] b, input cin, output [8:0] sum);
    wire [7:0] c_out; 
    wire [7:0] S;
    wire [7:0] d;


    fadder f1(d[0], sum[0], a[0], b[0], cin);
    hadder h1(d[1], S[0], a[1], b[1]);
    hadder h2(d[2], S[1], a[2], b[2]);
    hadder h3(d[3], S[2], a[3], b[3]);
    hadder h4(d[4], S[3], a[4], b[4]);
    hadder h5(d[5], S[4], a[5], b[5]);
    hadder h6(d[6], S[5], a[6], b[6]);
    hadder h7(d[7], S[6], a[7], b[7]);


    hadder h8(c_out[0], sum[1], S[0], d[0]);
    fadder f2(c_out[1], sum[2], S[1], d[1], c_out[0]);
    fadder f3(c_out[2], sum[3], S[2], d[2], c_out[1]);
    fadder f4(c_out[3], sum[4], S[3], d[3], c_out[2]);
    fadder f5(c_out[4], sum[5], S[4], d[4], c_out[3]);
    fadder f6(c_out[5], sum[6], S[5], d[5], c_out[4]);
    fadder f7(c_out[6], sum[7], S[6], d[6], c_out[5]);
    hadder h9(c_out[7], sum[8], c_out[6], d[7]);
endmodule

 
