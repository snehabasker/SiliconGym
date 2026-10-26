module mux2_w5(input [4:0] d0, input [4:0] d1, input sel, output [4:0] y);
  assign y = sel ? d1 : d0;
endmodule
