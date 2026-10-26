module shiftreg_w2(input clk, input rst, input si, output reg [1:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[0:0], si};
endmodule
