module gray_enc_w3(input [2:0] b, output [2:0] g);
  assign g = b ^ (b >> 1);
endmodule
