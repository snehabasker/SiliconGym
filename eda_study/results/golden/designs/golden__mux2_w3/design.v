module mux2_w3(input [2:0] d0, input [2:0] d1, input sel, output [2:0] y);
  assign y = sel ? d1 : d0;
endmodule
