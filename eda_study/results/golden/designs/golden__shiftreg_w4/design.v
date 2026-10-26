module shiftreg_w4(input clk, input rst, input si, output reg [3:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[2:0], si};
endmodule
