module mux2_w6(input [5:0] d0, input [5:0] d1, input sel, output [5:0] y);
  assign y = sel ? d1 : d0;
endmodule
