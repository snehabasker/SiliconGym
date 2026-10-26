module dff_sync_hi(input clk, input rst, input [3:0] d, output reg [3:0] q);
  always @(posedge clk)
    if (rst == 1'b1) q <= 4'd0;
    else q <= d;
endmodule
