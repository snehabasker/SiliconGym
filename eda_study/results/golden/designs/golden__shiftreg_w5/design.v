module shiftreg_w5(input clk, input rst, input si, output reg [4:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[3:0], si};
endmodule
