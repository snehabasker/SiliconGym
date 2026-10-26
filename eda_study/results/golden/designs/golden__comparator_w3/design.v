module comparator_w3(input [2:0] a, input [2:0] b, output eq, output lt, output gt);
  assign eq = a == b;
  assign lt = a < b;
  assign gt = a > b;
endmodule
