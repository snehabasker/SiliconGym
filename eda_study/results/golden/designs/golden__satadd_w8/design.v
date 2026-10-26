module satadd_w8(input [7:0] a, input [7:0] b, output [7:0] y);
  wire [8:0] t = a + b;
  assign y = t[8] ? {8{1'b1}} : t[7:0];
endmodule
