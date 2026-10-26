module parity_w2(input [1:0] in, output p);
  assign p = ^in;
endmodule
