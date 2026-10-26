module shiftreg_w6(input clk, input rst, input si, output reg [5:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[4:0], si};
endmodule
