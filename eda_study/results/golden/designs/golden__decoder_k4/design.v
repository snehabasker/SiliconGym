module decoder_k4(input [3:0] in, input en, output [15:0] out);
  assign out = en ? (16'd1 << in) : 16'd0;
endmodule
