module satadd_w6(input [5:0] a, input [5:0] b, output [5:0] y);
  wire [6:0] t = a + b;
  assign y = t[6] ? {6{1'b1}} : t[5:0];
endmodule
