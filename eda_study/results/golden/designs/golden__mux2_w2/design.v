module mux2_w2(input [1:0] d0, input [1:0] d1, input sel, output [1:0] y);
  assign y = sel ? d1 : d0;
endmodule
