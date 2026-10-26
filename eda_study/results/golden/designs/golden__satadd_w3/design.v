module satadd_w3(input [2:0] a, input [2:0] b, output [2:0] y);
  wire [3:0] t = a + b;
  assign y = t[3] ? {3{1'b1}} : t[2:0];
endmodule
