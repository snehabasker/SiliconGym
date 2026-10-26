module decoder_k1(input in, input en, output [1:0] out);
  assign out = en ? (2'd1 << in) : 2'd0;
endmodule
