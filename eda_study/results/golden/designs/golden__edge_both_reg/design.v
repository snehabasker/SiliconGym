module edge_both_reg(input clk, input rst, input sig, output reg pulse);
  reg prev;
  always @(posedge clk) prev <= rst ? 1'b0 : sig;
  always @(posedge clk) pulse <= rst ? 1'b0 : (sig ^ prev);
endmodule
