module parity_w5(input [4:0] in, output p);
  assign p = ^in;
endmodule
