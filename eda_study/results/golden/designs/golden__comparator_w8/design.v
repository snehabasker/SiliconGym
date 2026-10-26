module comparator_w8(input [7:0] a, input [7:0] b, output eq, output lt, output gt);
  assign eq = a == b;
  assign lt = a < b;
  assign gt = a > b;
endmodule
