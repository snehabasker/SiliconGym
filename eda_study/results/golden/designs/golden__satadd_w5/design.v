module satadd_w5(input [4:0] a, input [4:0] b, output [4:0] y);
  wire [5:0] t = a + b;
  assign y = t[5] ? {5{1'b1}} : t[4:0];
endmodule
