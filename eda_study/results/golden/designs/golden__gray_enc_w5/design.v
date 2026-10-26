module gray_enc_w5(input [4:0] b, output [4:0] g);
  assign g = b ^ (b >> 1);
endmodule
