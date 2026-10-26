module shiftreg_w3(input clk, input rst, input si, output reg [2:0] q);
  always @(posedge clk)
    if (rst) q <= 0;
    else q <= {q[1:0], si};
endmodule
