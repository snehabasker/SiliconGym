module decoder_k2(input [1:0] in, input en, output [3:0] out);
  assign out = en ? (4'd1 << in) : 4'd0;
endmodule
