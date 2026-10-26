module comparator_w6(input [5:0] a, input [5:0] b, output eq, output lt, output gt);
  assign eq = a == b;
  assign lt = a < b;
  assign gt = a > b;
endmodule
