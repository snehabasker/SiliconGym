module edge_both_comb(input clk, input rst, input sig, output pulse);
  reg prev;
  always @(posedge clk) prev <= rst ? 1'b0 : sig;
  assign pulse = sig ^ prev;
endmodule
