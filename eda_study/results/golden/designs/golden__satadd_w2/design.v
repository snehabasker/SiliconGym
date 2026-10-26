module satadd_w2(input [1:0] a, input [1:0] b, output [1:0] y);
  wire [2:0] t = a + b;
  assign y = t[2] ? {2{1'b1}} : t[1:0];
endmodule
