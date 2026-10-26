module parity_w4(input [3:0] in, output p);
  assign p = ^in;
endmodule
