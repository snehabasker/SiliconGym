module mux2_w8(input [7:0] d0, input [7:0] d1, input sel, output [7:0] y);
  assign y = sel ? d1 : d0;
endmodule
