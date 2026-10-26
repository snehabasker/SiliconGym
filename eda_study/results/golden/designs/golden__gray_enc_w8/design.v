module gray_enc_w8(input [7:0] b, output [7:0] g);
  assign g = b ^ (b >> 1);
endmodule
