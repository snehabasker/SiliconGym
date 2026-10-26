module comparator_w5(input [4:0] a, input [4:0] b, output eq, output lt, output gt);
  assign eq = a == b;
  assign lt = a < b;
  assign gt = a > b;
endmodule
