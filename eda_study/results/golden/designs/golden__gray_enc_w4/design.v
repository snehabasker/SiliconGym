module gray_enc_w4(input [3:0] b, output [3:0] g);
  assign g = b ^ (b >> 1);
endmodule
