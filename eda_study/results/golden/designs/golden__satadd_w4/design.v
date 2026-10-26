module satadd_w4(input [3:0] a, input [3:0] b, output [3:0] y);
  wire [4:0] t = a + b;
  assign y = t[4] ? {4{1'b1}} : t[3:0];
endmodule
