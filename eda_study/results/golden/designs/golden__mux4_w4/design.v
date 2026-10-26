module mux4_w4(input [3:0] d0, input [3:0] d1, input [3:0] d2, input [3:0] d3, input [1:0] sel, output [3:0] y);
  assign y = sel == 0 ? d0 : sel == 1 ? d1 : sel == 2 ? d2 : d3;
endmodule
