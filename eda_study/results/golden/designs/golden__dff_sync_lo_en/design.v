module dff_sync_lo_en(input clk, input rst_n, input en, input [3:0] d, output reg [3:0] q);
  always @(posedge clk)
    if (rst_n == 1'b0) q <= 4'd0;
    else if (en) q <= d;
endmodule
