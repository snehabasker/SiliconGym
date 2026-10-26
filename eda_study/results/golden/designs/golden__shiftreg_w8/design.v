module shiftreg_w8(input clk, input rst, input si, output reg [7:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[6:0], si};
endmodule
