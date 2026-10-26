module parity_w3(input [2:0] in, output p);
  assign p = ^in;
endmodule
