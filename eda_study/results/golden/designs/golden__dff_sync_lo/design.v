module dff_sync_lo(input clk, input rst_n, input [3:0] d, output reg [3:0] q);
  always @(posedge clk)
    if (rst_n == 1'b0) q <= 4'd0;
    else q <= d;
endmodule
