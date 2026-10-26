module parity_w8(input [7:0] in, output p);
  assign p = ^in;
endmodule
